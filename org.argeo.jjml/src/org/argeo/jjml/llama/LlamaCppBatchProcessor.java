package org.argeo.jjml.llama;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.IntBuffer;
import java.nio.channels.CompletionHandler;
import java.util.Arrays;
import java.util.List;
import java.util.Objects;
import java.util.StringJoiner;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.CompletionStage;

/**
 * Processing capabilities based on llama.cpp's batch API.
 * 
 * @see llama.h -- llama_batch
 */
public class LlamaCppBatchProcessor {
	final private LlamaCppModel model;
	final private LlamaCppContext context;

	private LlamaCppSamplerChain samplerChain;
	private LlamaCppNativeSampler validatingSampler;

	final private int NO_OUTPUT_ID;

	private int contextPosition = 0;

	public LlamaCppBatchProcessor(LlamaCppContext context, LlamaCppSamplerChain samplerChain) {
		this(context, samplerChain, null);
	}

	public LlamaCppBatchProcessor(LlamaCppContext context, LlamaCppSamplerChain samplerChain,
			LlamaCppNativeSampler validatingSampler) {
		Objects.requireNonNull(context);
		this.context = context;
		this.model = context.getModel();
		this.samplerChain = samplerChain;
		this.validatingSampler = validatingSampler;

		// there will never be an output id >= batch size
		this.NO_OUTPUT_ID = this.context.getBatchSize();
	}

	/*
	 * NATIVE METHODS
	 */
	private static native int doWriteBatch(long contextPointer, long samplerChainPointer, int contextPosition,
			IntBuffer[] input, int[] sequenceIds, int[] outputIds, boolean lastLogit);

	private static native int doReadBatch(long contextPointer, long samplerChainPointer, long grammarSamplerPointer,
			int contextPosition, IntBuffer[] output, int[] sequenceIds, int[] outputIds,
			CompletionHandler<Integer, Integer> completionHandler);

	/*
	 * LOW-LEVEL ACCESS
	 */
	synchronized void writeBatch(IntBuffer[] inputs, int[] sequenceIds, int[] outputIds, boolean lastLogits) {
		contextPosition = doWriteBatch(context.getAsLong(), samplerChain.getAsLong(), contextPosition, inputs,
				sequenceIds, outputIds, lastLogits);
		if (lastLogits && contextPosition > 0) {// end of user input
			samplerChain.reset();
			if (validatingSampler != null)
				validatingSampler.reset();
		}
	}

	synchronized void readBatch(IntBuffer[] outputs, int[] sequenceIds, int[] outputIds,
			CompletionHandler<Integer, Integer> completionHandler) {
		assert outputs.length == sequenceIds.length;
		contextPosition = doReadBatch(context.getAsLong(), samplerChain.getAsLong(),
				validatingSampler != null ? validatingSampler.getAsLong() : 0, contextPosition, outputs, sequenceIds,
				outputIds, completionHandler);
	}

	/*
	 * USABLE METHODS
	 */
	public String processSingleBatch(String systemPrompt, int sequenceId) {
		int[] sequenceIds = { sequenceId };
		return processBatch(systemPrompt, sequenceIds);
	}

	public String processBatch(String prompt, int[] sequenceIds) {
		return processBatch(prompt, sequenceIds, null, null);
	}

	public String processBatch(String prompt, int[] sequenceIds, String[] parameters, String postPrompt) {
		LlamaCppTokenList promptTL = model.tokenize(prompt, true);

//		int predictMax = context.getBatchSize();
		int parallelCount = sequenceIds.length;
//		int outputMax = predictMax - systemPromptTL.size();
		int outputMax = context.getBatchSize();
		int requiredContextSize = promptTL.size() + outputMax * parallelCount * 10;

		LlamaCppContext contextToUse = context;

		int contextSize = contextToUse.getContextSize();
//		System.out.println("Context size: " + contextSize);
		if (contextToUse.getContextSize() < requiredContextSize)
			throw new IllegalArgumentException(
					"The required KV cache size " + requiredContextSize + " is not big enough, only " + contextSize
							+ " available. Reduce parallel or increase context size.");

		int[] outputIds = new int[sequenceIds.length];
		Arrays.fill(outputIds, NO_OUTPUT_ID);

		// direct buffer area
		IntBuffer buf;
		{
//			ByteBuffer.allocateDirect(requiredContextSize * Integer.BYTES);// warm up
//			long begin = System.nanoTime();
			ByteBuffer directBuf = ByteBuffer.allocateDirect(requiredContextSize * Integer.BYTES);
			directBuf.order(ByteOrder.nativeOrder());// IMPORTANT!
//			long end = System.nanoTime();
//			System.out.println("Allocated buffer in    " + (end - begin) / 10 + " ns.");
			buf = directBuf.asIntBuffer();
		}

		int tokenCount = promptTL.size();
		int batchSize = context.getBatchSize();

		int batchCount = tokenCount / batchSize;
		if (tokenCount % batchSize != 0)
			batchCount = batchCount + 1;
		for (int i = 0; i < batchCount; i++) {
			IntBuffer input = buf.slice();
			boolean lastLogits;
			if (i == batchCount - 1) {
				input.limit(tokenCount % batchSize == 0 ? batchSize : tokenCount % batchSize);
				lastLogits = parameters == null;
			} else {
				input.limit(batchSize);
				lastLogits = false;
			}
			buf.position(buf.position() + input.limit());

			// copy data
			input.put(promptTL.getTokens(), i * batchSize, input.limit());
			input.flip();

			writeBatch(new IntBuffer[] { input }, sequenceIds, outputIds, lastLogits);
		}

		if (parameters != null) {
			if (parameters.length != parallelCount)
				throw new IllegalArgumentException("Parameters count different from sequence count");

			IntBuffer[] inputs = new IntBuffer[parallelCount];
			for (int i = 0; i < parallelCount; i++) {
				LlamaCppTokenList parameterTL = model.tokenize(parameters[i], true);
				if (parameterTL.size() * parallelCount > batchSize)// TODO be more precise / robust
					throw new IllegalArgumentException("Parameter '" + parameters[i] + "' is too long.");
				inputs[i] = buf.slice();
				inputs[i].limit(parameterTL.size());
				buf.position(buf.position() + inputs[i].limit());

				// copy data
				inputs[i].put(parameterTL.getTokens(), 0, inputs[i].limit());
				inputs[i].flip();
			}
			writeBatch(inputs, sequenceIds, outputIds, postPrompt == null);
		}

		if (postPrompt != null) {
			LlamaCppTokenList postPromptTL = model.tokenize(postPrompt, true);
			if (postPromptTL.size() > batchSize)// TODO be more precise / robust
				throw new IllegalArgumentException("Post prompt '" + postPrompt + "' is too long.");
			IntBuffer input = buf.slice();
			input.limit(postPromptTL.size());
			buf.position(buf.position() + input.limit());

			// copy data
			input.put(postPromptTL.getTokens(), 0, input.limit());
			input.flip();

			writeBatch(new IntBuffer[] { input }, sequenceIds, outputIds, true);
		}

		StringBuffer[] outputStrings = new StringBuffer[parallelCount];
		for (int i = 0; i < outputStrings.length; i++)
			outputStrings[i] = new StringBuffer();

//		int currSequenceCount = parallelCount;
//		int[] currSequenceIds = new int[currSequenceCount];
//		System.arraycopy(sequenceIds, 0, currSequenceIds, 0, currSequenceIds.length);
//		int[] currOutputIds = new int[currSequenceCount];
//		System.arraycopy(outputIds, 0, currOutputIds, 0, currOutputIds.length);
		boolean reading = true;
		reads: while (reading) {
			IntBuffer[] outputs = new IntBuffer[parallelCount];
			outputs: for (int i = 0; i < parallelCount; i++) {
				if (outputIds[i] == NO_OUTPUT_ID) {
					outputs[i] = null;
					continue outputs;
				}
//			IntBuffer output = buf.slice(buf.position(), outputMax); // Java 17
				IntBuffer output = buf.slice();
				output.limit(outputMax);
				outputs[i] = output;
				buf.position(buf.position() + output.limit());
			}

			long begin = System.nanoTime();
			CompletionHandler<Integer, Integer> completionHandler = new CompletionHandler<Integer, Integer>() {

				@Override
				public void failed(Throwable exc, Integer attachment) {
				}

				@Override
				public void completed(Integer result, Integer sequenceIndex) {
					System.out.println("Sequence with index " + sequenceIndex + " completed.");

					int sequenceId = sequenceIds[sequenceIndex];
					IntBuffer output = outputs[sequenceIndex];
					int outputId = outputIds[sequenceIndex];
					if (outputId == NO_OUTPUT_ID) {
						// generation completed
//						output.flip();
//						int[] newTokens = new int[output.limit() - output.position()];
//						output.get(newTokens);
//						LlamaCppTokenList newTL = new LlamaCppTokenList(model, newTokens);
//						String outputStr = newTL.getAsText();
//						res.add(outputStr);
					}
				}
			};

//		writeBatch(new IntBuffer[] { input }, sequenceIds, true);
			readBatch(outputs, sequenceIds, outputIds, completionHandler);

			long end = System.nanoTime();
			System.out.println("Read batch in    " + (end - begin) / 1 + " ns.");

			int sequencesLeft = 0;
			for (int i = 0; i < outputIds.length; i++) {
				IntBuffer output = outputs[i];
				if (output != null) {
					output.flip();
					int[] newTokens = new int[output.limit() - output.position()];
					output.get(newTokens);
					LlamaCppTokenList newTL = new LlamaCppTokenList(model, newTokens);
					String outputStr = newTL.getAsText();
//					System.out.println("\n\n\n Sequence " + i + "\n\n\n" + outputStr);
					outputStrings[i].append(outputStr);
					// res.add(outputStr);
				}

				if (outputIds[i] != NO_OUTPUT_ID) {
					sequencesLeft++;
				} else {

				}
			}

//			int[] newOutputIds = new int[currSequenceCount];
//			int[] newSequenceIds = new int[currSequenceCount];
//			int newIndex = 0;
//			for (int i = 0; i < currOutputIds.length; i++) {
//				if (currOutputIds[i] == NO_OUTPUT_ID) {
//					currSequenceCount--;
//				} else {
//					newOutputIds[newIndex] = currOutputIds[i];
//					newSequenceIds[newIndex] = currSequenceIds[i];
//					newIndex++;
//				}
//			}
//
//			if (currSequenceCount != newIndex)
//				throw new IllegalStateException("New index should be equald to curr seuqence count");

			if (sequencesLeft == 0)
				break reads;

			System.out.println(sequencesLeft + " sequences left");

			if (buf.position() + sequencesLeft * outputMax > buf.capacity()) {
				System.err.println("Main buffer will be full, aborting...");
				break reads;
			}

			// TODO check context size and break the loop
			// TODO timeout?

//			currSequenceIds = new int[currSequenceCount];
//			System.arraycopy(newSequenceIds, 0, currSequenceIds, 0, currSequenceIds.length);
//			currOutputIds = new int[currSequenceCount];
//			System.arraycopy(newOutputIds, 0, currOutputIds, 0, currOutputIds.length);
		}

//		for (int i = 0; i < outputs.length; i++) {
//			IntBuffer output = outputs[i];
//			output.flip();
//			int[] newTokens = new int[output.limit() - output.position()];
//			output.get(newTokens);
//			LlamaCppTokenList newTL = new LlamaCppTokenList(model, newTokens);
//			String outputStr = newTL.getAsText();
//			res.add(outputStr);
//		}
		StringJoiner res = new StringJoiner(
				"\n\n\n---------------------------------------------------------------\n\n\n");
		for (int i = 0; i < outputStrings.length; i++)
			res.add(outputStrings[i]);
		return res.toString();

	}

	/*
	 * Static utilities
	 */
	public static <T> CompletionStage<Object> anyOf(List<CompletionStage<T>> css) {
		return CompletableFuture
				.anyOf(css.stream().map(CompletionStage::toCompletableFuture).toArray(CompletableFuture[]::new));
	}

	public static <T> CompletionStage<Void> allOf(List<CompletionStage<T>> css) {
		return CompletableFuture
				.allOf(css.stream().map(CompletionStage::toCompletableFuture).toArray(CompletableFuture[]::new));
	}

}
