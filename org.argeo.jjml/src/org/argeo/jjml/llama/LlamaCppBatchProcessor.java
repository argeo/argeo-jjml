package org.argeo.jjml.llama;

import java.lang.reflect.Array;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.IntBuffer;
import java.nio.channels.CompletionHandler;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Objects;
import java.util.Set;
import java.util.StringJoiner;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.CompletionStage;

/**
 * A lightweight object coordinating the processing of multiple sequences.
 * 
 * @see llama.h - llama_batch
 */
public class LlamaCppBatchProcessor {
	private final LlamaCppContext context;
	private final LlamaCppVocabulary vocabulary;

	private LlamaCppSamplerChain samplerChain;
	private LlamaCppNativeSampler validatingSampler;

	/** Marker that end-of-generation has been reached for this sequence. */
	private final int NO_OUTPUT_ID;

	/** Current position from the context perspective. */
	private volatile int contextPosition = 0;

	// parallelism
	private final int parallelCount;
	private final /* const */ int[] sequenceIds;
	private final int[] outputIds;

	public LlamaCppBatchProcessor(LlamaCppContext context, LlamaCppSamplerChain samplerChain) {
		this(context, samplerChain, null, Collections.singleton(0));
	}

	public LlamaCppBatchProcessor(LlamaCppContext context, LlamaCppSamplerChain samplerChain,
			LlamaCppNativeSampler validatingSampler, Set<Integer> sequenceIds) {
		Objects.requireNonNull(context);
		Objects.requireNonNull(samplerChain);
		Objects.requireNonNull(sequenceIds);

		this.context = context;
		this.vocabulary = context.getModel().getVocabulary();
		this.samplerChain = samplerChain;
		this.validatingSampler = validatingSampler;

		// there will never be an output id >= batch size
		this.NO_OUTPUT_ID = this.context.getBatchSize();

		// parallelism
		if (sequenceIds.isEmpty())
			throw new IllegalArgumentException("There must be at least one sequence");
		this.parallelCount = sequenceIds.size();
		this.sequenceIds = new int[parallelCount];
		List<Integer> lst = new ArrayList<>(sequenceIds);
		Collections.sort(lst);// ensure predictable order, as a best practice
		for (int i = 0; i < lst.size(); i++)
			this.sequenceIds[i] = lst.get(i);
		this.outputIds = new int[parallelCount];
		Arrays.fill(outputIds, NO_OUTPUT_ID);
	}

	/*
	 * NATIVE METHODS
	 */
	private static native int doWrite(long contextPointer, long samplerChainPointer, int contextPosition,
			IntBuffer[] input, int[] offsets, int[] lengths, int[] sequenceIds, int[] outputIds, boolean lastLogit);

	private static native int doWriteArrays(long contextPointer, long samplerChainPointer, int contextPosition,
			int[][] input, int[] offsets, int[] lengths, int[] sequenceIds, int[] outputIds, boolean lastLogit);

	private static native int doRead(long contextPointer, long samplerChainPointer, long grammarSamplerPointer,
			int contextPosition, IntBuffer[] output, int[] offsets, int[] lengths, int[] sequenceIds, int[] outputIds,
			CompletionHandler<Integer, Integer> completionHandler);

	private static native int doReadToArrays(long contextPointer, long samplerChainPointer, long grammarSamplerPointer,
			int contextPosition, int[][] output, int[] offsets, int[] lengths, int[] sequenceIds, int[] outputIds,
			CompletionHandler<Integer, Integer> completionHandler);

	/*
	 * LOW-LEVEL ACCESS
	 */
	/**
	 * Convenience method for common input to all sequences, splitting the input in
	 * inputs of the batch size, and calling
	 * {@link #writeBatch(IntBuffer[], boolean)}.
	 */
	protected synchronized void writeBatch(IntBuffer buf, boolean thenLastLogits) {
		int tokenCount = buf.remaining();
		int batchSize = getContext().getBatchSize();
		int batchCount = tokenCount / batchSize;
		if (tokenCount % batchSize != 0)
			batchCount = batchCount + 1;
		for (int i = 0; i < batchCount; i++) {
			IntBuffer input = buf.slice();
			boolean lastLogits;
			if (i == batchCount - 1) {
				input.limit(tokenCount % batchSize == 0 ? batchSize : tokenCount % batchSize);
				lastLogits = thenLastLogits;
			} else {
				input.limit(batchSize);
				lastLogits = false;
			}
			buf.position(buf.position() + input.limit());
			writeBatch(new IntBuffer[] { input }, lastLogits);
		}

		// writeBatch(new IntBuffer[] { input }, lastLogits);
	}

	/**
	 * Write tokens to the context.
	 * 
	 * @param inputs     The inputs for each sequence, or a single input common to
	 *                   all sequences. The size of this array must be either one or
	 *                   {@link #getParallelCount()}.
	 * @param lastLogits Whether the last logits should be computed, thus completing
	 *                   the current write cycle.
	 * @throws IllegalArgumentException If the inputs count is different of one
	 *                                  (common to all sequences) or
	 *                                  {@link #getParallelCount()}.
	 */
	protected synchronized void writeBatch(IntBuffer[] inputs, boolean lastLogits) throws IllegalArgumentException {
		if (!(inputs.length == 1 || inputs.length == parallelCount))
			throw new IllegalArgumentException("There must be"
					+ (parallelCount > 1 ? " either one or " + parallelCount + " inputs" : " only one input"));
		int[] offsets = new int[inputs.length];
		int[] lengths = new int[inputs.length];
		int[][] arrays = new int[inputs.length][];
		boolean allDirect = areAllBuffersDirect(inputs, offsets, lengths);

		if (allDirect) {
			contextPosition = doWrite(context.getAsLong(), samplerChain.getAsLong(), contextPosition, inputs, offsets,
					lengths, sequenceIds, outputIds, lastLogits);
		} else {
			buffersToArrays(inputs, offsets, lengths, arrays, true);
			contextPosition = doWriteArrays(context.getAsLong(), samplerChain.getAsLong(), contextPosition, arrays,
					offsets, lengths, sequenceIds, outputIds, lastLogits);
		}

		if (lastLogits && contextPosition > 0) {// end of user input
			samplerChain.reset();
			if (validatingSampler != null)
				validatingSampler.reset();
		}
	}

	/**
	 * Convenience method when there is only one sequence, calling
	 * {@link #readBatchAsync(IntBuffer[], CompletableFuture[])}.
	 * 
	 * @throws UnsupportedOperationException If there is more than one sequence.
	 */
	protected CompletableFuture<Boolean> readBatchAsync(IntBuffer output) {
		if (getParallelCount() != 1)
			throw new UnsupportedOperationException(
					"There are " + getParallelCount() + " sequences, while only one is allowed");
		return readBatchAsync(new IntBuffer[] { output }, null);
	}

	/**
	 * Asynchronously read generated tokens from the context.
	 * 
	 * @param outputs             Where to write the tokens. It must be of size
	 *                            {@link #getParallelCount()}.
	 * @param generationCompleted an optional list of {@link CompletableFuture}
	 *                            which will be completed when the related
	 *                            individual sequence is completed. if the
	 *                            completion value is <code>true</code> it means
	 *                            that generation of this sequence was completed
	 *                            properly, that is an end-og-generation token was
	 *                            sampled. Can be <code>null</code>.
	 * @return A {@link CompletableFuture} which will complete when all sequences
	 *         have completed the reading of this batch. If the completion value is
	 *         <code>true</code>, it means that all sequences have completed
	 *         generation properly, that is, an end-of-generation was sampled.
	 * @throws IllegalArgumentException If the outputs count is different from
	 *                                  {@link #getParallelCount()}.
	 */
	protected CompletableFuture<Boolean> readBatchAsync(IntBuffer[] outputs,
			CompletableFuture<Boolean>[] generationCompleted) throws IllegalArgumentException {
		if (outputs.length != parallelCount)
			throw new IllegalArgumentException("There must be " + parallelCount + " outputs");
		if (generationCompleted != null && generationCompleted.length != parallelCount)
			throw new IllegalArgumentException("There must be " + parallelCount + " callbacks");
		int[] offsets = new int[outputs.length];
		int[] lengths = new int[outputs.length];
		boolean allDirect = areAllBuffersDirect(outputs, offsets, lengths);
		int[][] arrays = allDirect ? null : new int[outputs.length][];

		// this will be notified by each sequence when it is completed
		CompletionHandler<Integer, Integer> completionHandler = new CompletionHandler<Integer, Integer>() {

			@Override
			public void failed(Throwable exc, Integer sequenceIndex) {
				if (generationCompleted != null)
					generationCompleted[sequenceIndex].completeExceptionally(exc);
			}

			@Override
			public void completed(Integer result, Integer sequenceIndex) {
				IntBuffer output = outputs[sequenceIndex];
				if (arrays != null && !output.hasArray()) {
					output.put(arrays[sequenceIndex], 0, result);
				} else {
					output.position(output.position() + result);
				}
				if (generationCompleted != null) {
					int outputId = outputIds[sequenceIndex];
					// notify that generation is completed for this sequence
					generationCompleted[sequenceIndex].complete(outputId == NO_OUTPUT_ID);
				}
			}
		};

		// this will complete when all sequences have been completed
		CompletableFuture<Boolean> allCompleted = CompletableFuture.supplyAsync(() -> {
			// We synchronize in order to make sure there won't be other write or read
			// updating the state (contextPosition, output IDs etc.)
			synchronized (LlamaCppBatchProcessor.this) {
				if (allDirect) {
					contextPosition = doRead(context.getAsLong(), samplerChain.getAsLong(),
							validatingSampler != null ? validatingSampler.getAsLong() : 0, contextPosition, outputs,
							offsets, lengths, sequenceIds, outputIds, completionHandler);
				} else {
					buffersToArrays(outputs, offsets, lengths, arrays, false);
					contextPosition = doReadToArrays(context.getAsLong(), samplerChain.getAsLong(),
							validatingSampler != null ? validatingSampler.getAsLong() : 0, contextPosition, arrays,
							offsets, lengths, sequenceIds, outputIds, completionHandler);
				}

				// check whether generation is completed for all sequences
				boolean allGenerationCompleted = true;
				for (int i = 0; i < outputIds.length; i++) {
					if (NO_OUTPUT_ID != outputIds[i]) {
						allGenerationCompleted = false;
						break;
					}
				}
				return allGenerationCompleted;
			}
		});
		return allCompleted;
	}

	/**
	 * Common routine to fill check whether all buffers are direct. If all buffers
	 * are direct, the arrays will be filled with proper values, otherwise they will
	 * need to be processed again (via
	 * {@link #buffersToArrays(IntBuffer[], int[], int[], int[][], boolean)}.
	 */
	private boolean areAllBuffersDirect(IntBuffer[] buffers, int[] offsets, int[] lengths) {
		boolean allDirect = true;
		for (int i = 0; i < buffers.length; i++) {
			IntBuffer buf = buffers[i];
			if (buf == null) {
				offsets[i] = 0;
				lengths[i] = 0;
			} else if (buf.isDirect()) {
				offsets[i] = buf.position();
				lengths[i] = buf.remaining();
			} else {
				allDirect = false;
				break;
			}
		}
		return allDirect;
	}

	/**
	 * Common routine to fill arrays, to be used when not all buffers are direct.
	 */
	private void buffersToArrays(IntBuffer[] buffers, int[] offsets, int[] lengths, int[][] arrays, boolean input) {
		for (int i = 0; i < buffers.length; i++) {
			IntBuffer buf = buffers[i];
			if (buf == null) {
				offsets[i] = 0;
				lengths[i] = 0;
				arrays[i] = null;
			} else if (buf.hasArray()) {
				offsets[i] = buf.arrayOffset() + buf.position();
				lengths[i] = buf.remaining();
				arrays[i] = buf.array();
			} else {// new array
				offsets[i] = 0;
				lengths[i] = buf.remaining();
				int[] arr = new int[lengths[i]];
				if (input)
					buf.get(arr);
				arrays[i] = arr;
			}
		}
	}

	/*
	 * USABLE METHODS
	 */
	public String processSingleBatch(String systemPrompt) {
		return processBatch(systemPrompt);
	}

	public String processBatch(String prompt) {
		return processBatch(prompt, null, null);
	}

	public String processBatch(String prompt, String[] parameters, String postPrompt) {
		IntBuffer promptTokens = vocabulary.tokenize(prompt);
		assert promptTokens.position() == 0;
		int tokenCount = promptTokens.limit();
		int[] promptArr = promptTokens.array();

		int outputMax = context.getBatchSize();
		int requiredContextSize = tokenCount + outputMax * parallelCount * 10;

		int contextSize = context.getContextSize();
//		System.out.println("Context size: " + contextSize);
		if (context.getContextSize() < requiredContextSize)
			throw new IllegalArgumentException(
					"The required KV cache size " + requiredContextSize + " is not big enough, only " + contextSize
							+ " available. Reduce parallel or increase context size.");

		boolean direct = false;
		// direct buffer area
		IntBuffer buf;
		if (direct) {
			ByteBuffer directBuf = ByteBuffer.allocateDirect(requiredContextSize * Integer.BYTES);
			directBuf.order(ByteOrder.nativeOrder());// IMPORTANT!
			buf = directBuf.asIntBuffer();
		} else {
			buf = IntBuffer.allocate(requiredContextSize);
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

				writeBatch(new IntBuffer[] { input }, lastLogits);
			}

			if (parameters != null) {
				if (parameters.length != parallelCount)
					throw new IllegalArgumentException("Parameters count different from sequence count");

				IntBuffer[] inputs = new IntBuffer[parallelCount];
				for (int i = 0; i < parallelCount; i++) {
//					LlamaCppTokenList parameterTL = model.tokenizeAsArray(parameters[i], true);
					IntBuffer parametersTokens = vocabulary.tokenize(parameters[i]);
					if (parametersTokens.remaining() * parallelCount > batchSize)// TODO be more precise / robust
						throw new IllegalArgumentException("Parameter '" + parameters[i] + "' is too long.");
					inputs[i] = buf.slice();
					inputs[i].limit(parametersTokens.remaining());
					buf.position(buf.position() + inputs[i].limit());

					// copy data
					inputs[i].put(parametersTokens.array(), 0, inputs[i].limit());
					inputs[i].flip();
				}
				writeBatch(inputs, postPrompt == null);
			}

			if (postPrompt != null) {
//				LlamaCppTokenList postPromptTL = model.tokenizeAsArray(postPrompt, true);
				IntBuffer postPromptTokens = vocabulary.tokenize(postPrompt);
				if (postPromptTokens.remaining() > batchSize)// TODO be more precise / robust
					throw new IllegalArgumentException("Post prompt '" + postPrompt + "' is too long.");
				IntBuffer input = buf.slice();
				input.limit(postPromptTokens.remaining());
				buf.position(buf.position() + input.limit());

				// copy data
				input.put(postPromptTokens.array(), 0, input.limit());
				input.flip();

				writeBatch(new IntBuffer[] { input }, true);
			}
		} else {
			IntBuffer input = buf.slice();
			vocabulary.tokenize(prompt, input, true, true);
			buf.position(input.position());

			input.flip();
			writeBatch(new IntBuffer[] { input }, true);
		}

		StringBuffer[] outputStrings = new StringBuffer[parallelCount];
		for (int i = 0; i < outputStrings.length; i++)
			outputStrings[i] = new StringBuffer();

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

			CompletableFuture<Boolean>[] generationCompleted = newGenerationCompletableFutures();
			CompletableFuture<Boolean> allCompleted = readBatchAsync(outputs, generationCompleted);
			allCompleted.join();

			long end = System.nanoTime();
			// System.out.println("Read batch in " + (end - begin) / 1 + " ns.");

			int sequencesLeft = 0;
			for (int i = 0; i < outputIds.length; i++) {
				IntBuffer output = outputs[i];
				if (output != null) {
					output.flip();
					String outputStr = vocabulary.deTokenize(output);
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
	 * ACCESSORS
	 */

	/** The number of sequences being processed in parallel. */
	public int getParallelCount() {
		return parallelCount;
	}

	/** The context currently being exclusively used by this processor. */
	protected LlamaCppContext getContext() {
		return context;
	}

	protected LlamaCppModel getModel() {
		return context.getModel();
	}

	/*
	 * UTILITIES
	 */
	protected CompletableFuture<Boolean>[] newGenerationCompletableFutures() {
		@SuppressWarnings("unchecked") // required by type erasing
		CompletableFuture<Boolean>[] generationCompleted = createArr(CompletableFuture.class);
		for (int i = 0; i < generationCompleted.length; i++)
			generationCompleted[i] = new CompletableFuture<Boolean>();
		return generationCompleted;
	}

	/**
	 * Instantiate an array with generics. A separate method is required in order to
	 * capture the type.
	 */
	@SuppressWarnings("unchecked") // required by type erasing
	private <T> T[] createArr(Class<T> clz) {
		return (T[]) Array.newInstance(clz, getParallelCount());
	}

	/*
	 * STATIC UTILITIES
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
