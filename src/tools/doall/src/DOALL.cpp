/*
 * Copyright 2016 - 2021  Angelo Matni, Simone Campanoni
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

 * The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
 * IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */
#include "DOALL.hpp"
#include "DOALLTask.hpp"

namespace llvm::noelle{

DOALL::DOALL (
  Noelle &noelle
) :
    ParallelizationTechnique{noelle}
  , enabled{true}
  , taskDispatcher{nullptr}
  , n{noelle}
  {

  /*
   * Define the signature of the task, which will be invoked by the DOALL dispatcher.
   */
  auto tm = this->n.getTypesManager();
  auto funcArgTypes = ArrayRef<Type*>({
    tm->getVoidPointerType(),
    tm->getIntegerType(64),
    tm->getIntegerType(64),
    tm->getIntegerType(64)
  });
  this->taskSignature = FunctionType::get(tm->getVoidType(), funcArgTypes, false);

  /*
   * Fetch the dispatcher to use to jump to a parallelized DOALL loop.
   */
  this->taskDispatcher = this->n.getProgram()->getFunction("NOELLE_DOALLDispatcher");


  if (this->taskDispatcher == nullptr){
    this->enabled = false;
    if (this->verbose != Verbosity::Disabled) {
      errs() << "DOALL: WARNING: function NOELLE_DOALLDispatcher couldn't be found. DOALL is disabled\n";
    }
  }

  /*
   * Synchronization: SyncFunction
   */
  this->SyncFunction = this->n.getProgram()->getFunction("NOELLE_SyncUpParallelWorkers");

  return ;
}

bool DOALL::canBeAppliedToLoop (
  LoopDependenceInfo *LDI,
  Heuristics *h
) const {
  if (this->verbose != Verbosity::Disabled) {
    errs() << "DOALL: Checking if the loop is DOALL\n";
  }

  /*
   * Check if DOALL is enabled.
   */
  if (!this->enabled){
    return false;
  }

  /*
   * Fetch the loop structure.
   */
  auto loopStructure = LDI->getLoopStructure();

  /*
   * The loop must have one single exit path.
   */
  auto numOfExits = 0;
  for (auto bb : loopStructure->getLoopExitBasicBlocks()){

    /*
     * Fetch the last instruction before the terminator
     */
    auto terminator = bb->getTerminator();
    auto prevInst = terminator->getPrevNode();

    /*
     * Check if the last instruction is a call to a function that cannot return (e.g., abort()).
     */
    if (prevInst == nullptr){
      numOfExits++;
      continue ;
    }
    if (auto callInst = dyn_cast<CallInst>(prevInst)){
      auto callee = callInst->getCalledFunction();
      if (  true
            && (callee != nullptr)
            && (callee->getName() == "exit")
        ){
        continue ;
      }
    }
    numOfExits++;
  }
  if (numOfExits != 1){
    if (this->verbose != Verbosity::Disabled) {
      errs() << "DOALL:   More than 1 loop exit blocks\n";
    }
    return false;
  }

  /*
   * The loop must have all live-out variables to be reducable.
   */
  auto sccManager = LDI->getSCCManager();
  if (!sccManager->areAllLiveOutValuesReducable(LDI->environment)) {
    if (this->verbose != Verbosity::Disabled) {
      errs() << "DOALL:   Some live-out values are not reducable\n";
    }
    return false;
  }

  /*
   * The compiler must be able to remove loop-carried data dependences of all SCCs with loop-carried data dependences.
   */
  auto nonDOALLSCCs = DOALL::getSCCsThatBlockDOALLToBeApplicable(LDI, this->n);
  if (nonDOALLSCCs.size() > 0){
    if (this->verbose != Verbosity::Disabled) {
      for (auto scc : nonDOALLSCCs) {
        errs() << "DOALL:   We found an SCC of the loop that is non clonable and non commutative\n" ;
        if (this->verbose >= Verbosity::Maximal) {
          // scc->printMinimal(errs(), "DOALL:     ") ;
          // DGPrinter::writeGraph<SCC, Value>("not-doall-loop-scc-" + std::to_string(LDI->getID()) + ".dot", scc);
          errs() << "DOALL:     Loop-carried data dependences\n";
          sccManager->iterateOverLoopCarriedDataDependences(scc, [](DGEdge<Value> *dep) -> bool {
            auto fromInst = dep->getOutgoingT();
            auto toInst = dep->getIncomingT();
            errs() << "DOALL:       " << *fromInst << " ---> " << *toInst ;
            if (dep->isMemoryDependence()){
              errs() << " via memory\n";
            } else {
              errs() << " via variable\n";
            }
            return false;
              });
        }
      }
    }

    /*
     * There is at least one SCC that blocks DOALL to be applicable.
     */
    return false;
  }

  /*
   * The loop must have at least one induction variable.
   * This is because the trip count must be controlled by an induction variable.
   */
  auto loopGoverningIVAttr = LDI->getLoopGoverningIVAttribution();
  if (!loopGoverningIVAttr){
    if (this->verbose != Verbosity::Disabled) {
      errs() << "DOALL:   Loop does not have an induction variable to control the number of iterations\n";
    }
    return false;
  }

  /*
   * NOTE: Due to a limitation in our ability to chunk induction variables,
   * all induction variables must have step sizes that are loop invariant
   */
  auto IVManager = LDI->getInductionVariableManager();
  for (auto IV : IVManager->getInductionVariables(*loopStructure)) {
    if (IV->isStepValueLoopInvariant()) {
      continue;
    }
    if (this->verbose != Verbosity::Disabled) {
      errs() << "DOALL:  Loop has an induction variable with step size that is not loop invariant\n";
    }
    return false;
  }

  /*
   * Check if the final value of the induction variable is a loop invariant.
   */
  auto invariantManager = LDI->getInvariantManager();
  LoopGoverningIVUtility ivUtility(loopStructure, *IVManager, *loopGoverningIVAttr);
  auto &derivation = ivUtility.getConditionValueDerivation();
  for (auto I : derivation) {
    if (!invariantManager->isLoopInvariant(I)){
      if (this->verbose != Verbosity::Disabled) {
        errs() << "DOALL:  Loop has the governing induction variable that is compared against a non-invariant\n";
        errs() << "DOALL:     The non-invariant is = " << *I << "\n";
      }
      return false;
    }
  }

  /*
   * The loop is a DOALL one.
   */
  if (this->verbose != Verbosity::Disabled) {
    errs() << "DOALL:   The loop can be parallelized with DOALL\n" ;
  }
  return true;
}

bool DOALL::apply (
  LoopDependenceInfo *LDI,
  Heuristics *h
) {

  /*
   * Check if DOALL is enabled.
   */
  if (!this->enabled){
    return false;
  }

  /*
   * Fetch the headers.
   */
  auto loopStructure = LDI->getLoopStructure();
  auto loopHeader = loopStructure->getHeader();
  auto loopPreHeader = loopStructure->getPreHeader();

  /*
   * Fetch the loop function.
   */
  auto loopFunction = loopStructure->getFunction();

  /*
   * Fetch the environment of the loop.
   */
  auto loopEnvironment = LDI->environment;

  /*
   * Print the parallelization request.
   */
  if (this->verbose != Verbosity::Disabled) {
    errs() << "DOALL: Start the parallelization\n";
    errs() << "DOALL:   Number of threads to extract = " << LDI->getMaximumNumberOfCores() << "\n";
    errs() << "DOALL:   Chunk size = " << LDI->DOALLChunkSize << "\n";
  }

  /*
   * Generate an empty task for the parallel DOALL execution.
   */
  auto chunkerTask = new DOALLTask(this->taskSignature, *this->n.getProgram());
  this->addPredecessorAndSuccessorsBasicBlocksToTasks(LDI, { chunkerTask });
  this->numTaskInstances = LDI->getMaximumNumberOfCores();

  /*
   * Allocate memory for all environment variables
   */
  auto preEnvRange = loopEnvironment->getEnvIndicesOfLiveInVars();
  auto postEnvRange = loopEnvironment->getEnvIndicesOfLiveOutVars();
  std::set<int> nonReducableVars(preEnvRange.begin(), preEnvRange.end());
  std::set<int> reducableVars(postEnvRange.begin(), postEnvRange.end());
  this->initializeEnvironmentBuilder(LDI, nonReducableVars, reducableVars);

  /*
   * Clone loop into the single task used by DOALL
   */
  this->cloneSequentialLoop(LDI, 0);
  if (this->verbose >= Verbosity::Maximal) {
    errs() << "DOALL:  Cloned loop\n";
  }

  /*
   * Load all loop live-in values at the entry point of the task.
   */
  auto envUser = this->envBuilder->getUser(0);
  for (auto envIndex : loopEnvironment->getEnvIndicesOfLiveInVars()) {
    envUser->addLiveInIndex(envIndex);
  }
  for (auto envIndex : loopEnvironment->getEnvIndicesOfLiveOutVars()) {
    envUser->addLiveOutIndex(envIndex);
  }
  this->generateCodeToLoadLiveInVariables(LDI, 0);

  /*
   * HACK: For now, this must follow loading live-ins as this re-wiring overrides
   * the live-in mapping to use locally cloned memory instructions that are live-in to the loop
   */
  if (LDI->isOptimizationEnabled(LoopDependenceInfoOptimization::MEMORY_CLONING_ID)) {
    this->cloneMemoryLocationsLocallyAndRewireLoop(LDI, 0);
  }

  /*
   * Fix the data flow within the parallelized loop by redirecting operands of
   * cloned instructions to refer to the other cloned instructions. Currently,
   * they still refer to the original loop's instructions.
   */
  this->adjustDataFlowToUseClones(LDI, 0);
  if (this->verbose >= Verbosity::Maximal) {
    errs() << "DOALL:  Adjusted data flow\n";
  }

  /*
   * Handle the reduction variables.
   */
  this->setReducableVariablesToBeginAtIdentityValue(LDI, 0);

  /*
   * Add the jump to start the loop from within the task.
   */
  this->addJumpToLoop(LDI, chunkerTask);

  /*
   * Perform the iteration-chunking optimization
   */
  this->rewireLoopToIterateChunks(LDI);
  if (this->verbose >= Verbosity::Maximal) {
    errs() << "DOALL:  Rewired induction variables and reducible variables\n";
  }

  /*
   * Add the final return to the single task's exit block.
   */
  IRBuilder<> exitB(tasks[0]->getExit());
  exitB.CreateRetVoid();

  /*
   * Store final results to loop live-out variables. Note this occurs after
   * all other code is generated. Propagated PHIs through the generated
   * outer loop might affect the values stored
   */
  this->generateCodeToStoreLiveOutVariables(LDI, 0);

  if (this->verbose >= Verbosity::Maximal) {
    errs() << "DOALL:  Stored live outs\n";
  }

  this->addChunkFunctionExecutionAsideOriginalLoop(LDI, loopFunction, this->n);

  /*
   * Final printing.
   */
  if (this->verbose >= Verbosity::Maximal) {
    // loopFunction->print(errs() << "DOALL:  Final outside-loop code:\n" );
    // errs() << "\n";
    tasks[0]->getTaskBody()->print(errs() << "DOALL:  Final parallelized loop:\n");
    errs() << "\n";
    // SubCFGs execGraph(*chunkerTask->getTaskBody());
    // DGPrinter::writeGraph<SubCFGs, BasicBlock>("doalltask-loop" + std::to_string(LDI->getID()) + ".dot", &execGraph);
    // SubCFGs execGraph2(*loopFunction);
    // DGPrinter::writeGraph<SubCFGs, BasicBlock>("doall-loop-" + std::to_string(LDI->getID()) + "-function.dot", &execGraph);
  }
  if (this->verbose != Verbosity::Disabled) {
    errs() << "DOALL: Exit\n";
  }

  return true;
}

void DOALL::addChunkFunctionExecutionAsideOriginalLoop (
  LoopDependenceInfo *LDI,
  Function *loopFunction,
  Noelle &par
) {

  /*
   * Create the environment.
   */
  this->allocateEnvironmentArray(LDI);
  this->populateLiveInEnvironment(LDI);

  /*
   * Fetch the pointer to the environment.
   */
  auto envPtr = envBuilder->getEnvArrayInt8Ptr();

  /*
   * Fetch the number of cores
   */
  auto numCores = ConstantInt::get(par.int64, LDI->getMaximumNumberOfCores());

  /*
   * Fetch the chunk size.
   */
  auto chunkSize = ConstantInt::get(par.int64, LDI->DOALLChunkSize);

  /*
   * Call the function that incudes the parallelized loop.
   */
  IRBuilder<> doallBuilder(this->entryPointOfParallelizedLoop);
  auto doallCallInst = doallBuilder.CreateCall(this->taskDispatcher, ArrayRef<Value *>({
    tasks[0]->getTaskBody(),
    envPtr,
    numCores,
    chunkSize
  }));

  /*
   * Synchronization: build a map between original loop and the dispatcherInst
   */
  originalLS = LDI->getLoopStructure();

  ///*
  // * Synchronization: create a bit for this dispatch indicating whether it's synced
  // * create variable for numCores and memoryIdx
  // */
  //IRBuilder<> entryBuilder(loopFunction->getEntryBlock().getTerminator());
  //auto int1Ty = IntegerType::get(entryBuilder.getContext(), 1);
  //isSyncedAlloca = entryBuilder.CreateAlloca(int1Ty);
  //entryBuilder.CreateStore(ConstantInt::get(int1Ty, 1), isSyncedAlloca);
  //auto int32Ty = IntegerType::get(entryBuilder.getContext(), 32);
  //numCoresAlloca = entryBuilder.CreateAlloca(int32Ty);
  //auto int64Ty = IntegerType::get(entryBuilder.getContext(), 64);
  //memoryIdxAlloca = entryBuilder.CreateAlloca(int64Ty);

  ///*
  // * Synchronization: store 0 to isSynced after dispatch inst
  // */
  //doallBuilder.CreateStore(ConstantInt::get(int1Ty, 0), isSyncedAlloca);

  /*
   * Synchronization: store call to dispatcherinst and use in Parallelizer to insert synchronization calls
   */
  dispatcherInst = doallCallInst;

  numThreadsUsed = doallBuilder.CreateExtractValue(doallCallInst, (uint64_t)0);
  //doallBuilder.CreateStore(numThreadsUsed, numCoresAlloca);

  memoryIndex = doallBuilder.CreateExtractValue(doallCallInst, (uint64_t)1);
  //doallBuilder.CreateStore(memoryIndex, memoryIdxAlloca);

  /*
   * Propagate the last value of live-out variables to the code outside the parallelized loop.
   */
  auto latestBBAfterDOALLCall = this->propagateLiveOutEnvironment(LDI, numThreadsUsed, memoryIndex);

  /*
   * Jump to the unique successor of the loop.
   */
  IRBuilder<> afterDOALLBuilder{latestBBAfterDOALLCall};
  afterDOALLBuilder.CreateBr(this->exitPointOfParallelizedLoop);

  return ;
}

Value * DOALL::fetchClone (Value *original) const {
  auto task = this->tasks[0];
  if (isa<ConstantData>(original)) return original;

  if (task->isAnOriginalLiveIn(original)){
    return task->getCloneOfOriginalLiveIn(original);
  }

  assert(isa<Instruction>(original));
  auto iClone = task->getCloneOfOriginalInstruction(cast<Instruction>(original));
  assert(iClone != nullptr);
  return iClone;
}

void DOALL::addJumpToLoop (LoopDependenceInfo *LDI, Task *t){

  /*
   * Fetch the header within the task.
   */
  auto loopStructure = LDI->getLoopStructure();
  auto loopHeader = loopStructure->getHeader();
  auto headerClone = t->getCloneOfOriginalBasicBlock(loopHeader);

  /*
   * Add a jump to the loop within the task.
   */
  IRBuilder<> entryBuilder(t->getEntry());
  entryBuilder.CreateBr(headerClone);

  return ;
}

BasicBlock * DOALL::propagateLiveOutEnvironment (LoopDependenceInfo *LDI, Value *numberOfThreadsExecuted, Value *memoryIndex) {
  auto builder = new IRBuilder<>(this->entryPointOfParallelizedLoop);



  /*
   * Fetch the loop headers.
   */
  auto loopSummary = LDI->getLoopStructure();
  auto loopPreHeader = loopSummary->getPreHeader();
  auto f = loopSummary->getFunction();




  /*
   * Fetch the SCC manager.
   */
  auto sccManager = LDI->getSCCManager();

  /*
   * Collect reduction operation information needed to accumulate reducable variables after parallelization execution
   */
  std::unordered_map<int, int> reducableBinaryOps;
  std::unordered_map<int, Value *> initialValues;
  for (auto envInd : LDI->environment->getEnvIndicesOfLiveOutVars()) {
    auto isReduced = envBuilder->isReduced(envInd);
    if (!isReduced) continue;

    auto producer = LDI->environment->producerAt(envInd);
    auto producerSCC = sccManager->getSCCDAG()->sccOfValue(producer);
    auto producerSCCAttributes = sccManager->getSCCAttrs(producerSCC);

    /*
     * HACK: Need to get accumulator that feeds directly into producer PHI, not any intermediate one
     */
    auto firstAccumI = *(producerSCCAttributes->getAccumulators().begin());
    auto binOpCode = firstAccumI->getOpcode();
    reducableBinaryOps[envInd] = sccManager->accumOpInfo.accumOpForType(binOpCode, producer->getType());

    PHINode *loopEntryProducerPHI = this->fetchLoopEntryPHIOfProducer(LDI, producer);
    auto initValPHIIndex = loopEntryProducerPHI->getBasicBlockIndex(loopPreHeader);
    auto initialValue = loopEntryProducerPHI->getIncomingValue(initValPHIIndex);
    initialValues[envInd] = castToCorrectReducibleType(*builder, initialValue, producer->getType());
  }

  /*
   * Synchronization: add SyncFunction before reduction
   */
  //reductionPt = this->entryPointOfParallelizedLoop;
  isReduction = false;
  if(initialValues.size()){
    isReduction = true;
    //errs() << "SUSAN: creating reduction:" << *reductionPt << "\n";
    //reductionPt = CreateSynchronization(f, *builder, this->entryPointOfParallelizedLoop, nullptr, 1);
    //SyncFunctionInserted = true;
    //delete builder;
    //builder = new IRBuilder<>(reductionPt);
  }


  auto afterReductionB = this->envBuilder->reduceLiveOutVariables(
    this->entryPointOfParallelizedLoop,
    *builder,
    reducableBinaryOps,
    initialValues,
    numberOfThreadsExecuted);

  /*
   * Free the memory.
   */
  delete builder;

  /*
   * If reduction occurred, then all environment loads to propagate live outs need to be
   * inserted after the reduction loop
   */
  IRBuilder<> *afterReductionBuilder;
  if (afterReductionB->getTerminator()) {
    afterReductionBuilder->SetInsertPoint(afterReductionB->getTerminator());
  } else {
    afterReductionBuilder = new IRBuilder<>(afterReductionB);
  }

  for (int envInd : LDI->environment->getEnvIndicesOfLiveOutVars()) {
    auto prod = LDI->environment->producerAt(envInd);

    /*
     * NOTE(angelo): If the environment variable isn't reduced, it is held in allocated
     * memory that needs to be loaded from in order to retrieve the value
     */
    auto isReduced = envBuilder->isReduced(envInd);
    Value *envVar;
    if (isReduced) {
      envVar = envBuilder->getAccumulatedReducableEnvVar(envInd);
    } else {
      envVar = afterReductionBuilder->CreateLoad(envBuilder->getEnvVar(envInd));
    }

    for (auto consumer : LDI->environment->consumersOf(prod)) {
      if (auto depPHI = dyn_cast<PHINode>(consumer)) {
        depPHI->addIncoming(envVar, this->exitPointOfParallelizedLoop);
        continue;
      }
      prod->print(errs() << "Producer of environment variable:\t"); errs() << "\n";
      errs() << "Loop not in LCSSA!\n";
      abort();
    }
    /*
    * Synchronization: store locations of first use of liveouts outside of the parallel region
    */
    for (auto consumer : LDI->environment->consumersOf(prod))
          LiveOutUses.push_back(consumer);
  }




  /*
   * Free the memory.
   */
  delete afterReductionBuilder;

  return afterReductionB;
}


}
