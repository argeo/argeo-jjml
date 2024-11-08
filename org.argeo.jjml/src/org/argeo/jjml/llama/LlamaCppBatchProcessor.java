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
		IntBuffer promptTokens = model.getVocabulary().tokenize(prompt);
		assert promptTokens.position() == 0;
		int tokenCount = promptTokens.limit();
		int[] promptArr = promptTokens.array();

		int parallelCount = sequenceIds.length;
		int outputMax = context.getBatchSize();
		int requiredContextSize = tokenCount + outputMax * parallelCount * 10;

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
			ByteBuffer directBuf = ByteBuffer.allocateDirect(requiredContextSize * Integer.BYTES);
			directBuf.order(ByteOrder.nativeOrder());// IMPORTANT!
			buf = directBuf.asIntBuffer();
		}

		int batchSize = context.getBatchSize();

		boolean tokenList = true;

		if (tokenList) {
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
				input.put(promptArr, i * batchSize, input.limit());
				input.flip();

				writeBatch(new IntBuffer[] { input }, sequenceIds, outputIds, lastLogits);
			}

			if (parameters != null) {
				if (parameters.length != parallelCount)
					throw new IllegalArgumentException("Parameters count different from sequence count");

				IntBuffer[] inputs = new IntBuffer[parallelCount];
				for (int i = 0; i < parallelCount; i++) {
//					LlamaCppTokenList parameterTL = model.tokenizeAsArray(parameters[i], true);
					IntBuffer parametersTokens = model.getVocabulary().tokenize(parameters[i]);
					if (parametersTokens.remaining() * parallelCount > batchSize)// TODO be more precise / robust
						throw new IllegalArgumentException("Parameter '" + parameters[i] + "' is too long.");
					inputs[i] = buf.slice();
					inputs[i].limit(parametersTokens.remaining());
					buf.position(buf.position() + inputs[i].limit());

					// copy data
					inputs[i].put(parametersTokens.array(), 0, inputs[i].limit());
					inputs[i].flip();
				}
				writeBatch(inputs, sequenceIds, outputIds, postPrompt == null);
			}

			if (postPrompt != null) {
//				LlamaCppTokenList postPromptTL = model.tokenizeAsArray(postPrompt, true);
				IntBuffer postPromptTokens = model.getVocabulary().tokenize(postPrompt);
				if (postPromptTokens.remaining() > batchSize)// TODO be more precise / robust
					throw new IllegalArgumentException("Post prompt '" + postPrompt + "' is too long.");
				IntBuffer input = buf.slice();
				input.limit(postPromptTokens.remaining());
				buf.position(buf.position() + input.limit());

				// copy data
				input.put(postPromptTokens.array(), 0, input.limit());
				input.flip();

				writeBatch(new IntBuffer[] { input }, sequenceIds, outputIds, true);
			}
		} else {
			IntBuffer input = buf.slice();
			model.getVocabulary().tokenize(prompt, input, true, true);
			buf.position(input.position());

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
//					System.out.println("Sequence with index " + sequenceIndex + " completed.");

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
			//System.out.println("Read batch in    " + (end - begin) / 1 + " ns.");

			int sequencesLeft = 0;
			for (int i = 0; i < outputIds.length; i++) {
				IntBuffer output = outputs[i];
				if (output != null) {
					output.flip();
					int[] newTokens = new int[output.limit() - output.position()];
					output.get(newTokens);
					String outputStr = model.getVocabulary().deTokenize(IntBuffer.wrap(newTokens));
					outputStrings[i].append(outputStr);
				}

				if (outputIds[i] != NO_OUTPUT_ID) {
					sequencesLeft++;
				} else {

				}
			}

			if (sequencesLeft == 0)
				break reads;

			System.out.println(sequencesLeft + " sequences left");

			if (buf.position() + sequencesLeft * outputMax > buf.capacity()) {
				System.err.println("Main buffer will be full, aborting...");
				break reads;
			}

			// TODO check context size and break the loop
			// TODO timeout?
		}
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
