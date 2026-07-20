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

import org.argeo.jjml.llm.instruct.LlamaCppInstructFormatter;
import org.argeo.jjml.llm.instruct.LlamaCppInstructPart;
import org.argeo.jjml.llm.util.JinjaOsCallFormatter;
import org.argeo.jjml.llm.util.models.ChatMlFormatter;
import org.argeo.jjml.llm.util.models.Granite4Formatter;
import org.argeo.jjml.llm.util.models.Ministral3Formatter;
import org.argeo.jjml.llm.util.models.Olmo2Formatter;
import org.argeo.jjml.llm.util.models.Qwen3Formatter;

/** A processor based on chat messages. */
public class LlamaCppInstructProcessor extends LlamaCppBatchProcessor {
	private final LlamaCppVocabulary vocabulary;

	private final LlamaCppInstructFormatter instructFormatter;

	protected PrintStream debugPrompts = null;

	// TODO use tokens[] from LlamaCppBatchProcessor ?
	private final ByteBuffer tokenBuffer;

	public LlamaCppInstructProcessor(LlamaCppContext context, LlamaCppSamplerChain samplerChain,
			LlamaCppInstructFormatter instructFormatter) {
		super(context, samplerChain);
		this.vocabulary = context.getModel().getVocabulary();
		this.instructFormatter = instructFormatter != null ? instructFormatter : getDefaultInstructFormatter(context);

		tokenBuffer = ByteBuffer.allocateDirect(4 * Integer.BYTES);
		tokenBuffer.order(ByteOrder.nativeOrder());
	}

	public LlamaCppInstructProcessor(LlamaCppContext context, LlamaCppSamplerChain samplerChain) {
		this(context, samplerChain, null);
	}

	protected LlamaCppInstructFormatter getDefaultInstructFormatter(LlamaCppContext context) {
		LlamaCppInstructFormatter instructFormatter = null;
		if (System.getenv(JinjaOsCallFormatter.ENV_JJML_JINJA_PYTHON_SCRIPT) != null)
			return new JinjaOsCallFormatter(context.getModel().getMetadataChatTemplate());

		// TODO introduce extension mechanisms
		String modelArchitecture = context.getModel().getArchitecture();
		switch (modelArchitecture) {
		case "llama":
			instructFormatter = new ChatMlFormatter();
			break;
		case "qwen35":
			instructFormatter = new Qwen3Formatter();
			break;
		case "mistral3":
			instructFormatter = new Ministral3Formatter();
			break;
		case "granitehybrid":
		case "granite":
			instructFormatter = new Granite4Formatter();
			break;
		case "olmo2":
			instructFormatter = new Olmo2Formatter();
			break;
		}

		if (instructFormatter == null)
			instructFormatter = new LlamaCppNativeChatFormatter(context.getModel().getMetadataChatTemplate());
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

	public void write(LlamaCppInstructPart message) {
		Objects.requireNonNull(message);
		String prompt = instructFormatter.formatMessage(message);
		writeFormatted(prompt);
	}

	protected void writeFormatted(String prompt) {
		if (debugPrompts != null)
			debugPrompts.print(prompt);

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

//	public String nextToken() {
//		if (isGenerationCompleted(0))
//			return null;
//		ByteBuffer nativeBuf = ByteBuffer.allocateDirect(1 * Integer.BYTES);
//		nativeBuf.order(ByteOrder.nativeOrder());
//		IntBuffer output = nativeBuf.asIntBuffer();
//		// IntBuffer output = IntBuffer.allocate(1);
//
//		CompletableFuture<Boolean>[] generationCompleted = newGenerationCompletableFutures();
//		CompletableFuture<Boolean> allCompleted = readBatchAsync(new IntBuffer[] { output }, generationCompleted);
//		allCompleted.join();
//
//		output.flip();
//		String outputStr = vocabulary.deTokenize(output);
//		return outputStr;
//	}

	public String nextAnswer() {
		if (isGenerationCompleted(0))
			return null;
		IntBuffer output = getNextAnswerTokenBuffer();

		CompletableFuture<Boolean>[] generationCompleted = newGenerationCompletableFutures();
		CompletableFuture<Boolean> allCompleted = readBatchAsync(new IntBuffer[] { output }, generationCompleted);
		allCompleted.join();

		output.flip();
		String outputStr = vocabulary.deTokenize(output);
		return outputStr;
	}

	protected IntBuffer getNextAnswerTokenBuffer() {
		tokenBuffer.clear();
		IntBuffer output = tokenBuffer.asIntBuffer();
		return output;
	}

	public void readMessage(PrintStream out) throws IOException {
		out.flush();
		// FIXME deal properly with charset, esp. on Windows
		// Requires Android API level 33
		readMessage(new PrintWriter(out, false, StandardCharsets.UTF_8));
	}

	public void readMessage(Writer writer) throws IOException {
		// in case a generation prompt is needed
		StringBuilder generationPrompt = new StringBuilder();
		instructFormatter.appendGenerationPrompt(generationPrompt);
		if (generationPrompt.length() > 0)
			writeFormatted(generationPrompt.toString());

		boolean reading = true;
		reads: while (reading) {
//			String outputStr = nextToken();
			String outputStr = nextAnswer();
			if (outputStr == null)
				break reads;
			writer.write(outputStr);
			writer.flush();
		}
	}

	public void setDebugPrompts(PrintStream debugPrompts) {
		this.debugPrompts = debugPrompts;
	}

	public LlamaCppVocabulary getVocabulary() {
		return vocabulary;
	}

}
