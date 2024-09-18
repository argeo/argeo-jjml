package org.argeo.jjml.llama;

import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.CompletionStage;
import java.util.function.Function;

public class LlamaCppBatchProcessor {
	private LlamaCppModel model;

	native void doProcessBatch(CompletableFuture<?>[] callbacks, LlamaCppContext context, int[] systemPrompt,
			int[][] sequencePrompts, int predictMax);

	public List<CompletionStage<LlamaCppTokenList>> processBatch(String systemPrompt, List<String> sequencePrompt,
			int predictMax) {
		List<CompletionStage<LlamaCppTokenList>> res = new ArrayList<>(sequencePrompt.size());

		LlamaCppTokenList systemPromptTL = model.tokenize(systemPrompt, true);

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

		LlamaCppContext context = new LlamaCppContext();
		context.setModel(model);
		context.setContextSize(maxKvSize);
		context.setMaximumBatchSize(Math.max(predictMax, parallelCount));
		context.init();

		doProcessBatch(callbacks.toArray(new CompletableFuture<?>[callbacks.size()]), context,
				systemPromptTL.getTokens(), null, predictMax);

		context.destroy();
		return res;
	}

	public void setModel(LlamaCppModel model) {
		this.model = model;
	}

	/*
	 * Static utilities
	 */
	static <T> CompletionStage<Object> anyOf(List<CompletionStage<T>> css) {
		return CompletableFuture
				.anyOf(css.stream().map(CompletionStage::toCompletableFuture).toArray(CompletableFuture[]::new));
	}

	static <T> CompletionStage<Void> allOf(List<CompletionStage<T>> css) {
		return CompletableFuture
				.allOf(css.stream().map(CompletionStage::toCompletableFuture).toArray(CompletableFuture[]::new));
	}

	public static void main(String[] args) {
		String modelPathStr;
		// modelPathStr = "/srv/ai/models/Meta-Llama-3.1-8B-Instruct-Q4_0.gguf";
		modelPathStr = "/srv/ai/models/OLMo-7B-Instruct-Q8_0.gguf";
		// modelPathStr = "/srv/ai/models/tinyllama-1.1b-chat-v1.0.Q8_0.gguf";

		String systemPrompt = """
				Short implementation of a cycle detection algorithm in Java:
				
				""";

		Path modelPath = Paths.get(modelPathStr);

		LlamaCppBackend backend = new LlamaCppBackend();
		backend.init();

		LlamaCppModel model = new LlamaCppModel();
		model.setLocalPath(modelPath);
		model.init();

		LlamaCppBatchProcessor processor = new LlamaCppBatchProcessor();
		processor.setModel(model);
		List<String> sequencePrompts = new ArrayList<>();
		sequencePrompts.add("");
		List<CompletionStage<LlamaCppTokenList>> output = processor.processBatch(systemPrompt, sequencePrompts, 512);
		for (CompletionStage<LlamaCppTokenList> cd : output) {
			cd.thenAccept((tl) -> {
				System.out.println(tl.toString());
			});
		}
		allOf(output);

		model.destroy();
		backend.destroy();
	}

}
