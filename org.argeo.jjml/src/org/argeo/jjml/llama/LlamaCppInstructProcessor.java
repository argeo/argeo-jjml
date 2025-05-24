package org.argeo.jjml.llama;

import java.io.IOException;
import java.io.PrintStream;
import java.io.PrintWriter;
import java.io.Writer;
import java.nio.IntBuffer;
import java.nio.charset.StandardCharsets;
import java.util.concurrent.CompletableFuture;
import java.util.function.Supplier;

/** A processor based on chat messages. */
public class LlamaCppInstructProcessor extends LlamaCppBatchProcessor {
	private final LlamaCppVocabulary vocabulary;

	public LlamaCppInstructProcessor(LlamaCppContext context, LlamaCppSamplerChain samplerChain) {
		super(context, samplerChain);
		this.vocabulary = context.getModel().getVocabulary();
	}

	public void write(Supplier<String> role, String message) {
		write(new LlamaCppChatMessage(role, message));
	}

	public void write(String role, String message) {
		write(new LlamaCppChatMessage(role, message));
	}

	public void write(LlamaCppChatMessage message) {
		String prompt = getModel().formatChatMessages(message);
		writeFormatted(prompt);
	}

	protected void writeFormatted(String prompt) {
		IntBuffer promptTokens = vocabulary.tokenize(prompt);
		assert promptTokens.position() == 0;
		int tokenCount = promptTokens.limit();
		int[] promptArr = promptTokens.array();

		int outputMax = getContext().getBatchSize();

		// TODO check whether it makes sense (pattern was taken from llama.cpp code)
		int requiredContextSize = tokenCount + outputMax * getParallelCount() * 10;

		int contextSize = getContext().getContextSize();
//			System.out.println("Context size: " + contextSize);
		if (getContext().getContextSize() < requiredContextSize)
			throw new IllegalArgumentException(
					"The required KV cache size " + requiredContextSize + " is not big enough, only " + contextSize
							+ " available. Reduce parallel or increase context size.");
		IntBuffer buf = IntBuffer.allocate(requiredContextSize);

		int batchSize = getContext().getBatchSize();

		int batchCount = tokenCount / batchSize;
		if (tokenCount % batchSize != 0)
			batchCount = batchCount + 1;
		for (int i = 0; i < batchCount; i++) {
			IntBuffer input = buf.slice();
			boolean lastLogits;
			if (i == batchCount - 1) {
				input.limit(tokenCount % batchSize == 0 ? batchSize : tokenCount % batchSize);
				lastLogits = true;
			} else {
				input.limit(batchSize);
				lastLogits = false;
			}
			buf.position(buf.position() + input.limit());

			// copy data
			input.put(promptArr, i * batchSize, input.limit());
			input.flip();

//			long begin = System.nanoTime();
			writeBatch(new IntBuffer[] { input }, lastLogits);
//			long end = System.nanoTime();
//			System.out.println("Wrote batch in " + (end - begin) / 1000000 + " ms.");
		}
	}

	public void readMessage(PrintStream out) throws IOException {
		// FIXME deal properly with charset, esp. on Windows
		readMessage(new PrintWriter(out, false, StandardCharsets.UTF_8));
	}

	public void readMessage(Writer writer) throws IOException {

		boolean reading = true;
		reads: while (reading) {
			IntBuffer output = IntBuffer.allocate(1);

//			long begin = System.nanoTime();

			CompletableFuture<Boolean>[] generationCompleted = newGenerationCompletableFutures();
			CompletableFuture<Boolean> allCompleted = readBatchAsync(new IntBuffer[] { output }, generationCompleted);
			allCompleted.join();

//			long end = System.nanoTime();
			// System.out.println("Read batch in " + (end - begin) / 1000000 + " ms.");

			output.flip();
			String outputStr = vocabulary.deTokenize(output);
			writer.write(outputStr);
			writer.flush();

			if (isGenerationCompleted(0))
				break reads;
		}
	}
}
