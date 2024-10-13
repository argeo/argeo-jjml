package org.argeo.jjml.llama;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.IntBuffer;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.CompletionStage;
import java.util.function.Function;

public class LlamaCppBatchProcessor {
	private LlamaCppModel model;
	private LlamaCppContext context;

	/*
	 * NATIVE METHODS
	 */
	native void doProcessBatch(CompletableFuture<?>[] callbacks, LlamaCppContext context, int[] systemPrompt,
			int[][] sequencePrompts, int predictMax);

	native int doProcessSingleBatch(long contextPointer, ByteBuffer buf, int predictMax);

	/*
	 * USABLE METHODS
	 */
	public String processSingleBatch(String systemPrompt, int predictMax) {
		ByteBuffer buf = ByteBuffer.allocateDirect(predictMax * Integer.BYTES);
		buf.order(ByteOrder.nativeOrder());

		LlamaCppTokenList systemPromptTL = model.tokenize(systemPrompt, true);
		int[] tokens = systemPromptTL.getTokens();
		IntBuffer intBuf = buf.asIntBuffer();
		intBuf.put(tokens);
		buf.position(tokens.length * Integer.BYTES);
		buf.flip();

		int maxKvSize = systemPromptTL.size() + (predictMax - systemPromptTL.size());

		LlamaCppContext contextToUse;
		if (context == null) {
			LlamaCppContextParams contextParams = LlamaCppContextParams.defaultContextParams();
			contextToUse = new LlamaCppContext();
			contextToUse.setModel(model);
			contextParams.setContextSize(maxKvSize);
			contextParams.setMaxBatchSize(predictMax);
			contextToUse.init();
		} else {
			contextToUse = context;
		}

		doProcessSingleBatch(contextToUse.getPointer(), buf, predictMax);

		int newPosition = buf.position();
		int newLimit = buf.limit();
		System.out.println("newPosition=" + newPosition + ", newLimit=" + newLimit);
		intBuf.limit(newLimit / Integer.BYTES);
		intBuf.position(newPosition / Integer.BYTES);
		int[] newTokens = new int[intBuf.limit() - intBuf.position()];
		intBuf.get(newTokens);
		LlamaCppTokenList newTL = new LlamaCppTokenList(model, newTokens);

		return newTL.toString();
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
