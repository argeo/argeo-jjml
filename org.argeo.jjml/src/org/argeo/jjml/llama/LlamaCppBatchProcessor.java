package org.argeo.jjml.llama;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.IntBuffer;
import java.nio.channels.CompletionHandler;
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

	private int contextPosition = 0;
	private int nextRead = -1;

	public LlamaCppBatchProcessor(LlamaCppContext context) {
		Objects.requireNonNull(context);
		this.context = context;
		this.model = context.getModel();
	}

	public LlamaCppBatchProcessor(LlamaCppModel model, int requiredContextSize, int maxBatchSize) {
		LlamaCppContextParams contextParams = LlamaCppContextParams.defaultContextParams();
		contextParams.setContextSize(requiredContextSize);
		contextParams.setMaxBatchSize(maxBatchSize);
//			contextParams.setMaxBatchSize(Math.max(predictMax, parallelCount));
		LlamaCppContext contextToUse = new LlamaCppContext();
		contextToUse.setModel(model);
		contextToUse.init(contextParams);
		this.context = contextToUse;
		this.model = this.context.getModel();
	}

	/*
	 * NATIVE METHODS
	 */
	private static native int doWriteBatch(long contextPointer, int contextPosition, IntBuffer[] input,
			int[] sequenceIds, boolean lastLogit);

	private static native int doReadBatch(long contextPointer, int contextPosition, IntBuffer[] output,
			int[] sequenceIds, CompletionHandler<Integer, Integer> completionHandler);

	/*
	 * LOW-LEVEL ACCESS
	 */
	void writeBatch(IntBuffer input, int sequenceId) {
		// TODO optimize natively?
		writeBatch(new IntBuffer[] { input }, new int[] { sequenceId });
	}

	synchronized void writeBatch(IntBuffer[] inputs, int[] sequenceIds) {
		// doWriteBatch(context.getPointer(), 0, input, sequenceIds, lastLogit);
		if (inputs.length > 1)
			throw new UnsupportedOperationException("Multiple inputs is not yet supported");
		int written = doWriteBatch(context.getPointer(), contextPosition, inputs, sequenceIds, true);
//		contextPosition = contextPosition + written;
		contextPosition = written;
		nextRead = written;
	}

	synchronized void readBatch(IntBuffer[] outputs, int[] sequenceIds,
			CompletionHandler<Integer, Integer> completionHandler) {
		assert outputs.length == sequenceIds.length;
		int nextWrite = doReadBatch(context.getPointer(), nextRead, outputs, sequenceIds, completionHandler);
		contextPosition = contextPosition + (nextWrite - nextRead);
	}

	/*
	 * USABLE METHODS
	 */
	public String processSingleBatch(String systemPrompt, int sequenceId) {
		int[] sequenceIds = { sequenceId };
		return processBatch(systemPrompt, sequenceIds);
	}

	public String processBatch(String systemPrompt, int[] sequenceIds) {
		LlamaCppTokenList systemPromptTL = model.tokenize(systemPrompt, true);

		int predictMax = context.getBatchSize();
		int parallelCount = sequenceIds.length;
		int outputMax = predictMax - systemPromptTL.size();
		int requiredContextSize = systemPromptTL.size() + outputMax * parallelCount;
		IntBuffer buf;
		{
//			ByteBuffer.allocateDirect(requiredContextSize * Integer.BYTES);// warmup
//			long begin = System.nanoTime();
			ByteBuffer directBuf = ByteBuffer.allocateDirect(requiredContextSize * Integer.BYTES);
			directBuf.order(ByteOrder.nativeOrder());// IMPORTANT!
//			long end = System.nanoTime();
//			System.out.println("Allocated buffer in    " + (end - begin) / 10 + " ns.");
			buf = directBuf.asIntBuffer();
		}
//		IntBuffer input = buf.slice(buf.position(), systemPromptTL.size()); // Java 17
		IntBuffer input = buf.slice();
		input.limit(systemPromptTL.size());
		buf.position(buf.position() + input.limit());
		input.put(systemPromptTL.getTokens());

		IntBuffer[] outputs = new IntBuffer[parallelCount];
		for (int i = 0; i < parallelCount; i++) {
//			IntBuffer output = buf.slice(buf.position(), outputMax); // Java 17
			IntBuffer output = buf.slice();
			output.limit(outputMax);
			outputs[i] = output;
			buf.position(buf.position() + output.limit());
		}

		LlamaCppContext contextToUse = context;

		int contextSize = contextToUse.getContextSize();
//		System.out.println("Context size: " + contextSize);
		if (contextToUse.getContextSize() < requiredContextSize)
			throw new IllegalArgumentException(
					"The required KV cache size " + requiredContextSize + " is not big enough, only " + contextSize
							+ " available. Reduce parallel or increase context size.");

		StringJoiner res = new StringJoiner(
				"\n\n\n---------------------------------------------------------------\n\n\n");

//		boolean singleBatch = false;
//		if (singleBatch) {
//			doProcessSingleBatch(contextToUse.getPointer(), input, outputs[0]);
//		} else {
		long begin = System.nanoTime();
		CompletionHandler<Integer, Integer> completionHandler = new CompletionHandler<Integer, Integer>() {

			@Override
			public void failed(Throwable exc, Integer attachment) {
			}

			@Override
			public void completed(Integer result, Integer sequenceId) {
				System.out.println("Sequence " + sequenceId + " completed.");
//
//					Integer idx = null;
//					for (int i = 0; i < sequenceIds.length; i++) {
//						if (sequenceIds[i] == sequenceId) {
//							idx = i;
//							break;
//						}
//					}
//					Objects.requireNonNull(idx);// assert?
			}
		};

//		CompletableFuture<Void> doIt = CompletableFuture.runAsync( //
//				() -> writeBatch(new IntBuffer[] { input }, sequenceIds, true) //
//		).thenRunAsync(//
//				() -> readBatch(outputs, sequenceIds, completionHandler));
//		doIt.join();

		writeBatch(new IntBuffer[] { input }, sequenceIds);
		readBatch(outputs, sequenceIds, completionHandler);

		// System.out.println("newContextPosition=" + newContextPosition);

//			contextPosition = doWriteBatch(contextToUse.getPointer(), contextPosition, new IntBuffer[] { input },
//					sequenceIds, true);
//			doReadBatch(contextToUse.getPointer(), contextPosition, outputs, sequenceIds, completionHandler);
		long end = System.nanoTime();
		System.out.println("Processed batch in    " + (end - begin) / 1 + " ns.");
//		}

		for (int i = 0; i < outputs.length; i++) {
			IntBuffer output = outputs[i];
//			output.limit(output.capacity());
			output.flip();
//			if (i == outputs.length - 1) {
//				System.out.println("LAST");
//			}
			int[] newTokens = new int[output.limit() - output.position()];
//			System.err.println("Before get");
//			System.err.flush();
			output.get(newTokens);
			LlamaCppTokenList newTL = new LlamaCppTokenList(model, newTokens);

//			System.err.println("Before detoken");
//			System.err.flush();
//			try {
//				Thread.sleep(100);
//			} catch (InterruptedException e) {
//				// TODO Auto-generated catch block
//				e.printStackTrace();
//			}
			String outputStr = newTL.getAsText();
			res.add(outputStr);
		}
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
