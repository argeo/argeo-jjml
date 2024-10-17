package org.argeo.jjml.llama;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.IntBuffer;
import java.nio.channels.CompletionHandler;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.StringJoiner;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.CompletionStage;
import java.util.function.Function;

public class LlamaCppBatchProcessor {
	private LlamaCppModel model;
	private LlamaCppContext context;

	ByteBuffer directBuf ;
	/*
	 * NATIVE METHODS
	 */
	private native void doProcessBatch(CompletableFuture<?>[] callbacks, LlamaCppContext context, int[] systemPrompt,
			int[][] sequencePrompts, int predictMax);

	private native int doProcessSingleBatch(long contextPointer, IntBuffer input, IntBuffer output);

	private static native int doWriteBatch(long contextPointer, int contextPosition, IntBuffer[] input,
			int[] sequenceIds, boolean lastLogit);

	private static native int doReadBatch(long contextPointer, int contextPosition, IntBuffer[] output,
			int[] sequenceIds, CompletionHandler<Integer, Integer> completionHandler);

	/*
	 * LOW-LEVEL ACCESS
	 */
	void writeBatch(ByteBuffer[] input, int[] sequenceIds, boolean lastLogit) {
		// doWriteBatch(context.getPointer(), 0, input, sequenceIds, lastLogit);
	}

	/*
	 * USABLE METHODS
	 */
	public String processSingleBatch(String systemPrompt, int predictMax) {
		LlamaCppTokenList systemPromptTL = model.tokenize(systemPrompt, true);

		 int[] sequenceIds = { 579, 258, 123, 78, 12 };
		// int[] sequenceIds = { 579, 258, 123 };
		//int[] sequenceIds = { 579, 258 };
//		int[] sequenceIds = { 756 };

		int parallelCount = sequenceIds.length;
		int outputMax = predictMax - systemPromptTL.size();
		int requiredContextSize = systemPromptTL.size() + outputMax * parallelCount;
		IntBuffer buf;
		{
//			ByteBuffer.allocateDirect(requiredContextSize * Integer.BYTES);// warmup
//			long begin = System.nanoTime();
			directBuf = ByteBuffer.allocateDirect(requiredContextSize * Integer.BYTES);
			directBuf.order(ByteOrder.nativeOrder());// IMPORTANT!
//			long end = System.nanoTime();
//			System.out.println("Allocated buffer in    " + (end - begin) / 10 + " ns.");
			buf = directBuf.asIntBuffer();
		}
		IntBuffer input = buf.slice(buf.position(), systemPromptTL.size());
		buf.position(buf.position() + input.capacity());
		input.put(systemPromptTL.getTokens());

		IntBuffer[] outputs = new IntBuffer[parallelCount];
		for (int i = 0; i < parallelCount; i++) {
			IntBuffer output = buf.slice(buf.position(), outputMax);
			outputs[i] = output;
			buf.position(buf.position() + output.capacity());
		}

		LlamaCppContext contextToUse;
		if (context == null) {
			LlamaCppContextParams contextParams = LlamaCppContextParams.defaultContextParams();
			contextToUse = new LlamaCppContext();
			contextToUse.setModel(model);
			contextParams.setContextSize(requiredContextSize);
			contextParams.setMaxBatchSize(Math.max(predictMax, parallelCount));
			contextToUse.init(contextParams);
		} else {
			contextToUse = context;
		}

		int contextSize = contextToUse.getContextSize();
//		System.out.println("Context size: " + contextSize);
		if (contextToUse.getContextSize() < requiredContextSize)
			throw new IllegalArgumentException(
					"The required KV cache size " + requiredContextSize + " is not big enough, only " + contextSize
							+ " available. Reduce parallel or increase context size.");

		StringJoiner res = new StringJoiner(
				"\n\n\n---------------------------------------------------------------\n\n\n");

		boolean singleBatch = false;
		if (singleBatch) {
			doProcessSingleBatch(contextToUse.getPointer(), input, outputs[0]);
		} else {
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

			int contextPosition = 0;

//			CompletableFuture<Integer> doIt = CompletableFuture.supplyAsync( //
//					() -> doWriteBatch(contextToUse.getPointer(), contextPosition, new IntBuffer[] { input },
//							sequenceIds, true) //
//			).thenApplyAsync(//
//					(pos) -> doReadBatch(contextToUse.getPointer(), pos, outputs, sequenceIds, completionHandler));
//
//			int newContextPosition = doIt.join();
			// System.out.println("newContextPosition=" + newContextPosition);

			contextPosition = doWriteBatch(contextToUse.getPointer(), contextPosition, new IntBuffer[] { input },
					sequenceIds, true);
			doReadBatch(contextToUse.getPointer(), contextPosition, outputs, sequenceIds, completionHandler);
			long end = System.nanoTime();
			System.out.println("Processed batch in    " + (end - begin) / 1 + " ns.");

			buf.flip();
			for (int i = 0; i < buf.capacity(); i++) {
				int value = buf.get();
				if (value != 0)
					System.out.println(i + "\t" + value);
			}
			System.out.println("capacity=" + buf.capacity());
		}

		for (int i = 0; i < outputs.length; i++) {
			IntBuffer output = outputs[i];
//			output.limit(output.capacity());
			output.flip();
			if (i == outputs.length - 1) {
				System.out.println("LAST");
			}
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

	public List<CompletionStage<LlamaCppTokenList>> processBatch(String systemPrompt, List<String> sequencePrompt,
			int predictMax) {
		Objects.requireNonNull(systemPrompt, "System prompt must be provided.");

		List<CompletionStage<LlamaCppTokenList>> res = new ArrayList<>(sequencePrompt.size());

		LlamaCppTokenList systemPromptTL = model.tokenize(systemPrompt, true);

//		System.out.println(Arrays.toString(systemPrompt.toCharArray()));
//		System.out.println(Arrays.toString(systemPromptTL.toString().toCharArray()));
//		assert systemPrompt.equals(systemPromptTL.toString());

		// TODO process sequence-specific prompts
		int parallelCount = sequencePrompt.size();
		int maxKvSize = systemPromptTL.size() + (predictMax - systemPromptTL.size()) * parallelCount;

		List<CompletableFuture<int[]>> callbacks = new ArrayList<>();
		for (int i = 0; i < sequencePrompt.size(); i++) {
			CompletableFuture<int[]> callbackCF = new CompletableFuture<>();
			CompletionStage<LlamaCppTokenList> resCS = callbackCF
					.thenApplyAsync((Function<int[], LlamaCppTokenList>) (arr) -> {
						LlamaCppTokenList tokenList = new LlamaCppTokenList(model, arr);
						return tokenList;
					});
			res.add(resCS);
			callbacks.add(callbackCF);
		}

		LlamaCppContext contextToUse;
		if (context == null) {
			LlamaCppContextParams contextParams = LlamaCppContextParams.defaultContextParams();
			contextToUse = new LlamaCppContext();
			contextToUse.setModel(model);
			contextParams.setContextSize(maxKvSize);
			contextParams.setMaxBatchSize(Math.max(predictMax, parallelCount));
			contextToUse.init();
		} else {
			contextToUse = context;
		}

		doProcessBatch(callbacks.toArray(new CompletableFuture<?>[callbacks.size()]), contextToUse,
				systemPromptTL.getTokens(), null, predictMax);

		if (context == null)
			contextToUse.destroy();
		return res;
	}

	public void setModel(LlamaCppModel model) {
		this.model = model;
	}

	public void setContext(LlamaCppContext context) {
		// TODO check consistency with model
		this.context = context;
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
