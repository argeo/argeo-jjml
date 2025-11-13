
//!/usr/bin/env -S java -cp /usr/share/java/org.argeo.jjml.jar
import static java.lang.Boolean.FALSE;
import static java.lang.Boolean.parseBoolean;
import static java.lang.System.Logger.Level.INFO;
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
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.PrintStream;
import java.io.PrintWriter;
import java.lang.System.Logger;
import java.nio.IntBuffer;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.CompletionStage;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ForkJoinPool;
import java.util.concurrent.Future;
import java.util.concurrent.FutureTask;
import java.util.function.BiFunction;
import java.util.function.Consumer;

import org.argeo.jjml.llm.LlamaCppBackend;
import org.argeo.jjml.llm.LlamaCppBatchProcessor;
import org.argeo.jjml.llm.LlamaCppChatMessage;
import org.argeo.jjml.llm.LlamaCppContext;
import org.argeo.jjml.llm.LlamaCppModel;
import org.argeo.jjml.llm.LlamaCppNative;
import org.argeo.jjml.llm.LlamaCppSamplerChain;
import org.argeo.jjml.llm.LlamaCppSamplers;
import org.argeo.jjml.llm.LlamaCppVocabulary;
import org.argeo.jjml.llm.params.ContextParam;
import org.argeo.jjml.llm.params.ModelParam;
import org.argeo.jjml.llm.params.ModelParams;
import org.argeo.jjml.llm.util.SimpleModelDownload;
import org.argeo.jjml.llm.util.SimpleProgressCallback;

/** A minimal command line interface for batch processing and simple chat. */
public class JjmlDummyCli {
	private final static Logger logger = System.getLogger(JjmlDummyCli.class.getName());

	private final static String DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant.";

	/** Force chat mode in (Eclipse) IDE, when no proper console is available. */
	private final static boolean developing = parseBoolean(System.getProperty("JjmlDummyCli.ide", FALSE.toString()));

	public static void main(String... args) throws Exception {
		if (args.length == 0) {
			System.err.println("A GGUF model must be specified");
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
		String arg0 = args[0];
		Path modelPath = Paths.get(arg0);
		if (!Files.exists(modelPath))
			modelPath = new SimpleModelDownload().getOrDownloadModel(arg0, new SimpleProgressCallback());
		if (!Files.exists(modelPath))
			throw new IllegalArgumentException("Could not find GGUF model " + modelPath);

		String systemPrompt = DEFAULT_SYSTEM_PROMPT;
		if (args.length > 1) {
			systemPrompt = args[1];
			if (systemPrompt.contains(File.separator) || systemPrompt.contains("/")) {
				try {// try to interpret as file
					systemPrompt = Files.readString(Paths.get(systemPrompt), UTF_8);
				} catch (IOException e) {
					System.err.println("Could not interpret '" + systemPrompt + "' as a file, using it as value...");
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
			modelParams = modelParams.with(n_gpu_layers, 99);
		}

		logger.log(INFO, "Loading model " + modelPath + " ...");
		Future<LlamaCppModel> loaded = LlamaCppModel.loadAsync(modelPath, modelParams, new SimpleProgressCallback(),
				null);
		try (LlamaCppModel model = loaded.get(); //
				LlamaCppContext context = new LlamaCppContext(model, defaultContextParams()); //
		) {
			SimpleChat processor = new SimpleChat(systemPrompt, context);
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
							processor.apply(input, (str) -> {
								out.print(str);
								out.flush();
							}).thenAccept((v) -> {
								out.print("\n> ");
								out.flush();
							});
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
					processor.apply(input, (str) -> {
						System.out.print(str);
						System.out.flush();
					}).toCompletableFuture().join();
				}
			} finally {
				// make sure that we are not reading before cleaning up backend
				processor.cancelCurrentRead();
			}
		}
	}

	private static void printUsage(PrintStream out) {
		out.println("Usage: java " + JjmlDummyCli.class.getName() //
				+ " <path/to/model.gguf | hf/repo > [<system prompt>]");

		out.println();
		out.println("- Opens a basic interactive chat when in a terminal.");
		out.println("- Piping input will disable interactivity and submit the whole input as a single user prompt.");
		out.println("- The context does not auto-extend, that is, it will be full at some point.");
		out.println("- All external inputs should be encoded with UTF-8.");
		out.println("- If <system prompt> contains a file separator or /, it will be loaded as a file.");
		out.println("- <system prompt> default is '" + DEFAULT_SYSTEM_PROMPT + "'.");
		out.println("- If <system prompt> is set to \"\", message formatting with chat template is disabled.");

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
		out.println("-D" + LlamaCppNative.SYSTEM_PROPERTY_LIBPATH_JJML_LLM + "=");
		out.println("-D" + LlamaCppNative.SYSTEM_PROPERTY_LIBPATH_LLAMACPP + "=");
		out.println("-D" + LlamaCppNative.SYSTEM_PROPERTY_LIBPATH_GGML + "=");
		out.println();
		out.println("#");
		out.println("# WARNING - This is a suboptimal informational implementation.");
		out.println("# JJML is meant to be used directly as a Java library.");
		out.println("#");
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
