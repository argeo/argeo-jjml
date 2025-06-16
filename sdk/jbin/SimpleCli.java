
import static java.lang.Boolean.FALSE;
import static java.lang.Boolean.parseBoolean;
import static java.nio.charset.StandardCharsets.UTF_8;
import static org.argeo.jjml.llm.LlamaCppContext.defaultContextParams;
import static org.argeo.jjml.llm.LlamaCppNative.ENV_GGML_CUDA_ENABLE_UNIFIED_MEMORY;
import static org.argeo.jjml.llm.params.ModelParam.n_gpu_layers;
import static org.argeo.jjml.llm.util.InstructRole.ASSISTANT;
import static org.argeo.jjml.llm.util.InstructRole.SYSTEM;
import static org.argeo.jjml.llm.util.InstructRole.USER;

import java.io.BufferedReader;
import java.io.Console;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.PrintStream;
import java.io.PrintWriter;
import java.nio.IntBuffer;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.StringJoiner;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.CompletionStage;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ForkJoinPool;
import java.util.concurrent.FutureTask;
import java.util.function.BiFunction;
import java.util.function.Consumer;
import java.util.function.Function;
import java.util.function.Supplier;

import org.argeo.jjml.llm.LlamaCppBackend;
import org.argeo.jjml.llm.LlamaCppBatchProcessor;
import org.argeo.jjml.llm.LlamaCppChatMessage;
import org.argeo.jjml.llm.LlamaCppContext;
import org.argeo.jjml.llm.LlamaCppEmbeddingProcessor;
import org.argeo.jjml.llm.LlamaCppModel;
import org.argeo.jjml.llm.LlamaCppNative;
import org.argeo.jjml.llm.LlamaCppSamplerChain;
import org.argeo.jjml.llm.LlamaCppSamplers;
import org.argeo.jjml.llm.LlamaCppVocabulary;
import org.argeo.jjml.llm.params.ContextParam;
import org.argeo.jjml.llm.params.ModelParam;
import org.argeo.jjml.llm.params.ModelParams;
import org.argeo.jjml.llm.params.PoolingType;

/** A minimal command line interface for batch processing and simple chat. */
public class SimpleCli {
	private final static String DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant.";

	/** Force chat mode in (Eclipse) IDE, when no proper console is available. */
	final static boolean developing = parseBoolean(System.getProperty("SimpleCli.ide", FALSE.toString()));

	public static void main(String... args) throws Exception {
		if (args.length == 0) {
			System.err.println("A model must be specified");
			printUsage(System.err);
			System.exit(1);
		}
		if ("--help".equals(args[0])) {
			printUsage(System.out);
			System.exit(0);
		}
		/*
		 * ARGUMENTS
		 */
		Path modelPath = Paths.get(args[0]);
		if (!Files.exists(modelPath))
			throw new FileNotFoundException("Model " + modelPath + " does not exist");

		boolean embeddings = Boolean.parseBoolean(System.getProperty(ContextParam.embeddings.asSystemProperty()));
		int chunkSize = 0;
		String embeddingsFormat = "csv";
		String systemPrompt = DEFAULT_SYSTEM_PROMPT;
		if (embeddings) {
			if (args.length > 1) {
				chunkSize = Integer.parseInt(args[1]);
				if (args.length > 2) {
					embeddingsFormat = args[2];
				}
			}
		} else {
			if (args.length > 1) {
				systemPrompt = args[1];
				if (systemPrompt.contains(File.separator) || systemPrompt.contains("/")) {
					try {// try to interpret as file
						systemPrompt = Files.readString(Paths.get(systemPrompt), UTF_8);
					} catch (IOException e) {
						System.err
								.println("Could not interpret '" + systemPrompt + "' as a file, using it as value...");
					}
				}
			}
		}

		/*
		 * AUTOCONFIG
		 */
		ModelParams modelParams = LlamaCppModel.defaultModelParams();
		if ("1".equals(System.getenv(ENV_GGML_CUDA_ENABLE_UNIFIED_MEMORY)) //
				&& System.getProperty(n_gpu_layers.asSystemProperty()) == null //
				&& LlamaCppBackend.supportsGpuOffload() //
				&& modelParams.n_gpu_layers() == 0 //
		) {
			// we assume we want as many layers offloaded as possible
			modelParams = modelParams.with(n_gpu_layers, 1024);
		}

		try (LlamaCppModel model = LlamaCppModel.load(modelPath, modelParams); //
				LlamaCppContext context = new LlamaCppContext(model, defaultContextParams()); //
		) {
			Object processor;
			if (embeddings) {
				processor = new SimpleEmbedding(context, chunkSize);
			} else {
				processor = new SimpleChat(systemPrompt, context);
			}

			Console console = System.console();
			final boolean isConsoleTerminal = console != null;
			// From Java 22, it will be:
			// boolean interactive = console.isTerminal();
			final boolean interactive = developing || isConsoleTerminal;

			try {
				if (interactive) {
					PrintWriter out = console != null ? console.writer() : new PrintWriter(System.out, true);
					out.print("> ");
					out.flush();
					try (BufferedReader reader = new BufferedReader(
							console != null ? console.reader() : new InputStreamReader(System.in))) {
						String line;
						while ((line = reader.readLine()) != null) {
							String input = handleHereDocument(line, reader);
							if (processor instanceof SimpleChat) {
								((SimpleChat) processor).apply(input, (str) -> {
									out.print(str);
									out.flush();
								}).thenAccept((v) -> {
									out.print("\n> ");
									out.flush();
								});
							} else if (processor instanceof SimpleEmbedding) {
								float[][] res = ((SimpleEmbedding) processor).apply(input);
								printEmbeddings(out, res, embeddingsFormat);
								out.print("\n> ");
								out.flush();
							}
						}
					}
				} else {// batch
					String input;
					try (BufferedReader in = new BufferedReader(new InputStreamReader(System.in, UTF_8))) {
						StringBuilder sb = new StringBuilder();
						final int BUFFER_SIZE = 4 * 1024;
						char[] buf = new char[BUFFER_SIZE];
						int numCharsRead;
						while ((numCharsRead = in.read(buf, 0, buf.length)) != -1)
							sb.append(buf, 0, numCharsRead);
						input = sb.toString();
					}
					if (processor instanceof SimpleChat) {
						((SimpleChat) processor).apply(input, (str) -> {
							System.out.print(str);
							System.out.flush();
						}).toCompletableFuture().join();
					} else if (processor instanceof SimpleEmbedding) {
						float[][] res = ((SimpleEmbedding) processor).apply(input);
						printEmbeddings(new PrintWriter(System.out, true, StandardCharsets.UTF_8), res,
								embeddingsFormat);
					}
				}
			} finally {
				// make sure that we are not reading before cleaning up backend
				if (processor instanceof SimpleChat)
					((SimpleChat) processor).cancelCurrentRead();
			}
		}
	}

	private static void printUsage(PrintStream out) {
		out.println("Usage: java " + SimpleCli.class.getName() //
				+ " <path/to/model.gguf> [<system prompt>]");
		out.println("Usage: java -Djjml.llama.context.embeddings=true " + SimpleCli.class.getName() //
				+ " <path/to/model.gguf> [<chunk size>] [ csv | pgvector ]");

		out.println();
		out.println("- Opens a basic interactive chat when in a terminal.");
		out.println("- Piping input will disable interactivity and submit the whole input as a single user prompt.");
		out.println("- The context does not auto-extend, that is, it will be full at some point.");
		out.println("- All external inputs should be encoded with UTF-8.");
		out.println("- If <system prompt> contains a file separator or /, it will be loaded as a file.");
		out.println("- <system prompt> default is '" + DEFAULT_SYSTEM_PROMPT + "'.");
		out.println("- If <system prompt> is set to \"\", message formatting with chat template is disabled.");
		out.println("- For embeddings, a <chunk size> of 0 (default) disable chunking.");
		out.println("- For embeddings, default output format is 'csv', while 'pgvector' generates VALUES.");

		out.println();
		out.println("# In interactive mode, use <<EOF for multi-line input. For example:");
		out.println();
		out.println("> Suggest improvements to this Java code: <<EOF");
		out.println("public static void main(String[] args) {");
		out.println("  System.out.println(\"Hello world!\");");
		out.println("}");
		out.println("EOF");

		out.println();
		out.println("# System properties for supported parameters (see llama.h for details):");
		out.println();
		for (ModelParam param : ModelParam.values())
			out.println("-D" + param.asSystemProperty() + "=");
		for (ContextParam param : ContextParam.values())
			out.println("-D" + param.asSystemProperty() + "=");

		out.println();
		out.println("# System properties for explicit paths to shared libraries:");
		out.println();
		out.println("-D" + LlamaCppNative.SYSTEM_PROPERTY_LIBPATH_JJML_LLAMA + "=");
		out.println("-D" + LlamaCppNative.SYSTEM_PROPERTY_LIBPATH_LLAMACPP + "=");
		out.println("-D" + LlamaCppNative.SYSTEM_PROPERTY_LIBPATH_GGML + "=");
		out.println();
		out.println("# WARNING: This is a suboptimal implementation. JJML is meant to be used as a Java library.");
	}

	/**
	 * Read a portion of the stream as a single stream based on a <<EOF delimiter.
	 */
	private static String handleHereDocument(String line, BufferedReader reader) throws IOException {
		int hereIndex = line.indexOf("<<");
		if (hereIndex < 0)
			return line;
		String kept = line.substring(0, hereIndex);
		String delimiter = line.substring(hereIndex + 2);
		StringBuilder sb = new StringBuilder(kept);
		if ("".equals(delimiter)) {// corner case, just add next line
			if ((line = reader.readLine()) != null)
				sb.append(line);
		} else {
			delimiter = delimiter.strip().split("\\s+")[0];
			here_doc: while ((line = reader.readLine()) != null) {
				if (line.strip().equals(delimiter))
					break here_doc;
				sb.append(line);
				sb.append('\n');
			}
		}
		return sb.toString();
	}

	/** The float array in a usable format. */
	private static void printEmbeddings(PrintWriter out, float[][] embeddings, String format) {
		if ("csv".equals(format))
			printEmbeddings(out, embeddings, "\n", () -> new StringJoiner(","));
		else if ("pgvector".equals(format))
			printEmbeddings(out, embeddings, ",\n", () -> new StringJoiner(",", "('[", "]')"));
		else
			throw new IllegalArgumentException("Unknown output format " + format);
	}

	/** Format a float array. */
	private static void printEmbeddings(PrintWriter out, float[][] embeddings, String vecorSep,
			Supplier<StringJoiner> valueSj) {
		for (int i = 0; i < embeddings.length; i++) {
			if (i != 0)
				out.print(vecorSep);
			StringJoiner sj = valueSj.get();
			for (int j = 0; j < embeddings[i].length; j++)
				sj.add(Float.toString(embeddings[i][j]));
			out.print(sj);
		}
	}
}

/**
 * A simple implementation of a chat system based on the low-level components.
 */
class SimpleChat extends LlamaCppBatchProcessor implements BiFunction<String, Consumer<String>, CompletionStage<Void>> {
	private final LlamaCppVocabulary vocabulary;

	private volatile boolean reading = false;
	private FutureTask<Void> currentRead = null;

	private final LlamaCppChatMessage systemMsg;
	private boolean firstMessage = true;
	private final boolean formatMessages;

	/*
	 * In llama.cpp examples/main, a new user prompt is obtained via a substring of
	 * the previous messages. We reproduce this behavior here, even though it is not
	 * clear yet whether it is useful.
	 */
	// TODO: check in details and remove this as it would greatly simplify the code
	private final boolean usePreviousMessages;
	private final List<LlamaCppChatMessage> messages;

	public SimpleChat(String systemPrompt, LlamaCppContext context) {
		this(systemPrompt, context, LlamaCppSamplers.newDefaultSampler(true));
	}

	/**
	 * Creates a simple chat processor.
	 * 
	 * @param systemPrompt The system prompt. If <code>null</code> or empty, it
	 *                     disables chat templating.
	 */
	public SimpleChat(String systemPrompt, LlamaCppContext context, LlamaCppSamplerChain samplerChain) {
		super(context, samplerChain);
		vocabulary = getModel().getVocabulary();
		if (systemPrompt == null || "".equals(systemPrompt)) {
			systemMsg = null;
			formatMessages = false;
			usePreviousMessages = false;
		} else {
			systemMsg = new LlamaCppChatMessage(SYSTEM, systemPrompt);
			formatMessages = true;
			usePreviousMessages = true;
		}
		messages = usePreviousMessages ? new ArrayList<>() : null;
	}

	@Override
	public CompletionStage<Void> apply(String message, Consumer<String> consumer) {
		if (currentRead != null && !currentRead.isDone()) {
			// throw new ConcurrentModificationException("Currently interacting, use
			// cancel.");
			cancelCurrentRead();
			if (message.trim().equals(""))
				return CompletableFuture.completedStage(null); // this was just for interruption
		}
//		message = message.replace("\\\n", "\n");
		String prompt;
		if (formatMessages) {
			LlamaCppChatMessage userMsg = new LlamaCppChatMessage(USER, message);
			if (usePreviousMessages) {
				String previousPrompts = messages.size() == 0 ? "" : getModel().formatChatMessages(messages);
				if (firstMessage) {
					if (systemMsg != null)
						messages.add(systemMsg);
					firstMessage = false;
				}
				messages.add(userMsg);
				String newPrompts = getModel().formatChatMessages(messages);
				assert previousPrompts.length() < newPrompts.length();
				prompt = newPrompts.substring(previousPrompts.length(), newPrompts.length());
			} else {
				List<LlamaCppChatMessage> lst = new ArrayList<>();
				if (firstMessage) {
					lst.add(systemMsg);
					firstMessage = false;
				}
				lst.add(userMsg);
				prompt = getModel().formatChatMessages(lst);
			}
		} else {
			prompt = message;
		}

		// tokenize
		IntBuffer input = vocabulary.tokenize(prompt);
		writeBatch(input, true);
		FutureTask<Void> future = new FutureTask<>(() -> {
			String reply = readAll(consumer);
			if (usePreviousMessages) {
				LlamaCppChatMessage assistantMsg = new LlamaCppChatMessage(ASSISTANT, reply);
				messages.add(assistantMsg);
			}
			return null;
		});
		setCurrentRead(future);
		ForkJoinPool.commonPool().execute(future);
		return CompletableFuture.runAsync(() -> {
			try {
				future.get();
			} catch (InterruptedException | ExecutionException e) {
				// TODO deal with it
			}
		});
	}

	protected String readAll(Consumer<String> consumer) {
		try {
			StringBuffer sb = new StringBuffer();
			reading = true;
			running: while (reading) {
				IntBuffer output = IntBuffer.allocate(getContext().getBatchSize());
				CompletableFuture<Boolean> done = SimpleChat.this.readBatchAsync(output);
				boolean generationCompleted = done.join();
				output.flip();
				String str = vocabulary.deTokenize(output);
				consumer.accept(str);
				if (usePreviousMessages)
					sb.append(str);

				output.clear();
				if (generationCompleted) {// generation completed as expected
					break running;
				}
				if (Thread.interrupted()) {// generation was interrupted
					int endOfGenerationToken = getContext().getModel().getEndOfGenerationToken();
					IntBuffer input = IntBuffer.allocate(1);
					input.put(endOfGenerationToken);
					input.flip();
					writeBatch(input, false);
					if (usePreviousMessages)
						sb.append(vocabulary.deTokenize(input));
					consumer.accept("");// flush
					break running;
				}
			}
			return sb.toString();
		} finally {
			reading = false;
		}
	}

	protected boolean isReading() {
		return reading;
	}

	protected void cancelCurrentRead() {
		if (currentRead == null)
			return;
		if (!currentRead.isDone()) {
			currentRead.cancel(true);
			while (isReading()) // wait for reading to complete
				try {
					Thread.sleep(100);
				} catch (InterruptedException e) {
					return;
				}
		}
	}

	private void setCurrentRead(FutureTask<Void> currentRead) {
		this.currentRead = currentRead;
	}
}

/** Computes embeddings based on chunks of a given size. */
class SimpleEmbedding extends LlamaCppEmbeddingProcessor implements Function<String, float[][]> {
	private final LlamaCppVocabulary vocabulary;

	private final int chunkSize;

	/**
	 * Constructor.
	 * 
	 * @param context   The context used to initialize this processor.
	 * @param chunkSize The size of the chunks. If <=0, the strings will be
	 *                  processed as a whole.
	 */
	public SimpleEmbedding(LlamaCppContext context, int chunkSize) {
		super(context);
		this.vocabulary = getContext().getModel().getVocabulary();
		this.chunkSize = chunkSize;
	}

	@Override
	public float[][] apply(String str) {
		if (chunkSize <= 0 || PoolingType.LLAMA_POOLING_TYPE_NONE.equals(getContext().getPoolingType())) {
			return processEmbeddings(Collections.singletonList(str));
		}
		int totalLength = str.length();
		IntBuffer[] inputs = new IntBuffer[totalLength / chunkSize + (totalLength % chunkSize == 0 ? 0 : 1)];
		for (int i = 0; i < inputs.length; i++) {
			String chunk;
			if (i == inputs.length - 1) {
				chunk = str.substring(i * chunkSize);
			} else {
				chunk = str.substring(i * chunkSize, (i + 1) * chunkSize);
			}
			inputs[i] = vocabulary.tokenize(chunk);
		}
		return processEmbeddings(inputs);
	}

}
