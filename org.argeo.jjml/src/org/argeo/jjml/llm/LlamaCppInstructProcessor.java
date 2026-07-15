package org.argeo.jjml.llm;

import java.io.IOException;
import java.io.PrintStream;
import java.io.PrintWriter;
import java.io.Writer;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.IntBuffer;
import java.nio.charset.StandardCharsets;
import java.util.Objects;
import java.util.concurrent.CompletableFuture;
import java.util.function.Supplier;

import org.argeo.jjml.llm.util.JinjaOsCallFormatter;

/** A processor based on chat messages. */
public class LlamaCppInstructProcessor extends LlamaCppBatchProcessor {
	private final LlamaCppVocabulary vocabulary;

	private final LlamaCppInstructFormatter instructFormatter;

	public LlamaCppInstructProcessor(LlamaCppContext context, LlamaCppSamplerChain samplerChain,
			LlamaCppInstructFormatter instructFormatter) {
		super(context, samplerChain);
		this.vocabulary = context.getModel().getVocabulary();
		this.instructFormatter = instructFormatter;
	}

	public LlamaCppInstructProcessor(LlamaCppContext context, LlamaCppSamplerChain samplerChain) {
		this(context, samplerChain, getDefaultInstructFormatter(context));
	}

	private static LlamaCppInstructFormatter getDefaultInstructFormatter(LlamaCppContext context) {
		// FIXME implement cleaner defaults
		LlamaCppInstructFormatter instructFormatter = System
				.getenv(JinjaOsCallFormatter.ENV_JJML_JINJA_PYTHON_SCRIPT) == null ? //
						new LlamaCppNativeChatFormatter(context.getModel().getMetadataChatTemplate()) //
						: new JinjaOsCallFormatter(context.getModel().getMetadataChatTemplate());

		// instructFormatter = new Ministral3InstructFormatter();
		return instructFormatter;
	}

	public void write(Supplier<String> role, String message) {
		Objects.requireNonNull(message);
		write(new LlamaCppChatMessage(role, message));
	}

	public void write(String role, String message) {
		Objects.requireNonNull(message);
		write(new LlamaCppChatMessage(role, message));
	}

	public void write(LlamaCppChatMessage message) {
		Objects.requireNonNull(message);
		String prompt = instructFormatter.formatChatMessages(message);
		writeFormatted(prompt);
	}

	protected void writeFormatted(String prompt) {
		IntBuffer promptTokens = vocabulary.tokenize(prompt);
		assert promptTokens.position() == 0;
		int tokenCount = promptTokens.limit();
		int[] promptArr = promptTokens.array();

		int outputMax = getContext().getBatchSize();

		// TODO check whether it makes sense (pattern was taken from llama.cpp code)
		int requiredContextSize = tokenCount + outputMax * getParallelCount();

		int remainingContextSize = getRemainingContextSize();
		if (remainingContextSize < requiredContextSize)
			throw new IllegalArgumentException(
					"The required remaining context size " + requiredContextSize + " is not big enough, only "
							+ remainingContextSize + " available. Reduce parallel or increase context size.");

		ByteBuffer nativeBuf = ByteBuffer.allocateDirect(requiredContextSize * Integer.BYTES);
		nativeBuf.order(ByteOrder.nativeOrder());
		IntBuffer buf = nativeBuf.asIntBuffer();
		// IntBuffer buf = IntBuffer.allocate(requiredContextSize);

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

			writeBatch(new IntBuffer[] { input }, lastLogits);
		}
	}

	public String nextToken() {
		if (isGenerationCompleted(0))
			return null;
		ByteBuffer nativeBuf = ByteBuffer.allocateDirect(1 * Integer.BYTES);
		nativeBuf.order(ByteOrder.nativeOrder());
		IntBuffer output = nativeBuf.asIntBuffer();
		// IntBuffer output = IntBuffer.allocate(1);

		CompletableFuture<Boolean>[] generationCompleted = newGenerationCompletableFutures();
		CompletableFuture<Boolean> allCompleted = readBatchAsync(new IntBuffer[] { output }, generationCompleted);
		allCompleted.join();

		output.flip();
		String outputStr = vocabulary.deTokenize(output);
		return outputStr;
	}

	public void readMessage(PrintStream out) throws IOException {
		out.flush();
		// FIXME deal properly with charset, esp. on Windows
		// Requires Android API level 33
		readMessage(new PrintWriter(out, false, StandardCharsets.UTF_8));
	}

	public void readMessage(Writer writer) throws IOException {

		boolean reading = true;
		reads: while (reading) {
			String outputStr = nextToken();
			if (outputStr == null)
				break reads;
			writer.write(outputStr);
			writer.flush();
		}
	}
}
