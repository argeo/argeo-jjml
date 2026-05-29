import static java.lang.Boolean.FALSE;
import static java.lang.Boolean.parseBoolean;
import static java.lang.System.Logger.Level.INFO;
import static java.nio.charset.StandardCharsets.UTF_8;
import static org.argeo.jjml.llm.LlamaCppContext.defaultContextParams;
import static org.argeo.jjml.llm.LlamaCppNative.ENV_GGML_CUDA_ENABLE_UNIFIED_MEMORY;
import static org.argeo.jjml.llm.params.ModelParam.n_gpu_layers;

import java.io.BufferedReader;
import java.io.Console;
import java.io.File;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.PrintStream;
import java.io.PrintWriter;
import java.lang.System.Logger;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.concurrent.Future;

import org.argeo.jjml.llm.LlamaCppBackend;
import org.argeo.jjml.llm.LlamaCppContext;
import org.argeo.jjml.llm.LlamaCppInstructProcessor;
import org.argeo.jjml.llm.LlamaCppModel;
import org.argeo.jjml.llm.LlamaCppNative;
import org.argeo.jjml.llm.LlamaCppSamplerChain;
import org.argeo.jjml.llm.LlamaCppSamplers;
import org.argeo.jjml.llm.params.ContextParam;
import org.argeo.jjml.llm.params.ModelParam;
import org.argeo.jjml.llm.params.ModelParams;
import org.argeo.jjml.llm.util.InstructRole;
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
			throw new IllegalArgumentException("Could not find GGUF model " + modelPath);

		String systemPrompt = DEFAULT_SYSTEM_PROMPT;
		Path systemPromptFile = null;
		if (args.length > 1) {
			systemPrompt = args[1];
			if (systemPrompt.contains(File.separator) || systemPrompt.contains("/")) {
				try {// try to interpret as file
					Path p = Paths.get(systemPrompt);
					systemPrompt = Files.readString(p, UTF_8);
					systemPromptFile = p;
				} catch (IOException e) {
					// ignore and use as string
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
				LlamaCppContext context = new LlamaCppContext(model, defaultContextParams() //
						.with(ContextParam.n_ctx, Math.min(model.getContextTrainingSize(), 20480)) //
						.with(ContextParam.n_threads, Runtime.getRuntime().availableProcessors()) //
				); //
				LlamaCppSamplerChain samplerChain = LlamaCppSamplers.newDefaultSampler(); //
		) {
			LlamaCppInstructProcessor processor = new LlamaCppInstructProcessor(context, samplerChain);
			Console console = System.console();
			final boolean isConsoleTerminal = console != null;
			// From Java 22, it will be:
			// boolean interactive = console.isTerminal();
			final boolean interactive = developing || isConsoleTerminal;

			// Initial context
			if (systemPromptFile != null) {
				Path initialStateFile = Paths.get(systemPromptFile.getFileName() + "."
						+ model.getMetadata().get("general.architecture") + ".ggsn");
				if (Files.exists(initialStateFile) && Files.getLastModifiedTime(initialStateFile)
						.compareTo(Files.getLastModifiedTime(systemPromptFile)) > 0) {
					processor.loadStateFile(initialStateFile);
					System.err.println("Loaded state file " + initialStateFile);
				} else {
					processor.write(InstructRole.SYSTEM, systemPrompt);
					processor.saveStateFile(initialStateFile);
					System.err.println("Created state file " + initialStateFile);
				}
			} else {
				processor.write(InstructRole.SYSTEM, systemPrompt);
			}

			// Processing
			if (interactive) {
				PrintWriter out = console != null ? console.writer() : new PrintWriter(System.out, true);
				out.print("> ");
				out.flush();
				try (BufferedReader reader = new BufferedReader(
						console != null ? console.reader() : new InputStreamReader(System.in))) {
					String line;
					while ((line = reader.readLine()) != null) {
						String input = handleHereDocument(line, reader);
						processor.write(InstructRole.USER, input);
						String nextToken;
						while ((nextToken = processor.nextToken()) != null) {
							out.print(nextToken);
							out.flush();
						}
						out.print("\n> ");
						out.flush();
					}
				}
			} else {// batch
				// input
				try (BufferedReader in = new BufferedReader(new InputStreamReader(System.in, UTF_8))) {
					StringBuilder sb = new StringBuilder();
					final int BUFFER_SIZE = 4 * 1024;
					char[] buf = new char[BUFFER_SIZE];
					int numCharsRead;
					while ((numCharsRead = in.read(buf, 0, buf.length)) != -1)
						sb.append(buf, 0, numCharsRead);
					processor.write(InstructRole.USER, sb.toString());
				}

				// output
				String nextToken;
				while ((nextToken = processor.nextToken()) != null) {
					System.out.print(nextToken);
				}
				System.out.flush();
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
		out.println("- All inputs and outputs should be encoded with UTF-8 (aka. chcp 65001 on Windows).");
		out.println("- If <system prompt> contains a file separator or /, it will be loaded as a file.");
		out.println("- If <system prompt> is loaded as a file, the context will be cached based on file timestamp.");
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
