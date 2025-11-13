
//!/usr/bin/env -S java -cp /usr/share/java/org.argeo.jjml.jar
import static java.lang.Boolean.FALSE;
import static java.lang.Boolean.parseBoolean;
import static java.lang.System.Logger.Level.INFO;
import static java.nio.charset.StandardCharsets.UTF_8;
import static org.argeo.jjml.llm.LlamaCppContext.defaultContextParams;
import static org.argeo.jjml.llm.LlamaCppNative.ENV_GGML_CUDA_ENABLE_UNIFIED_MEMORY;
import static org.argeo.jjml.llm.params.ModelParam.n_gpu_layers;

import java.io.BufferedReader;
import java.io.Console;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.PrintStream;
import java.io.PrintWriter;
import java.lang.System.Logger;
import java.nio.IntBuffer;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Collections;
import java.util.StringJoiner;
import java.util.concurrent.Future;
import java.util.function.Function;
import java.util.function.Supplier;

import org.argeo.jjml.llm.LlamaCppBackend;
import org.argeo.jjml.llm.LlamaCppContext;
import org.argeo.jjml.llm.LlamaCppEmbeddingProcessor;
import org.argeo.jjml.llm.LlamaCppModel;
import org.argeo.jjml.llm.LlamaCppNative;
import org.argeo.jjml.llm.LlamaCppVocabulary;
import org.argeo.jjml.llm.params.ContextParam;
import org.argeo.jjml.llm.params.ModelParam;
import org.argeo.jjml.llm.params.ModelParams;
import org.argeo.jjml.llm.params.PoolingType;
import org.argeo.jjml.llm.util.SimpleModelDownload;
import org.argeo.jjml.llm.util.SimpleProgressCallback;

/** A minimal command line interface for batch processing and simple chat. */
public class JjmlEmbeddings {
	private final static Logger logger = System.getLogger(JjmlEmbeddings.class.getName());

	/** Force chat mode in (Eclipse) IDE, when no proper console is available. */
	private final static boolean developing = parseBoolean(System.getProperty("JjmlEmbeddings.ide", FALSE.toString()));

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

		System.setProperty(ContextParam.embeddings.asSystemProperty(), "true");

		int chunkSize = 0;
		String embeddingsFormat = "csv";
		if (args.length > 1) {
			chunkSize = Integer.parseInt(args[1]);
			if (args.length > 2) {
				embeddingsFormat = args[2];
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
			SimpleEmbedding processor = new SimpleEmbedding(context, chunkSize);

			Console console = System.console();
			final boolean isConsoleTerminal = console != null;
			// From Java 22, it will be:
			// boolean interactive = console.isTerminal();
			final boolean interactive = developing || isConsoleTerminal;

			if (interactive) {
				PrintWriter out = console != null ? console.writer() : new PrintWriter(System.out, true);
				out.print("> ");
				out.flush();
				try (BufferedReader reader = new BufferedReader(
						console != null ? console.reader() : new InputStreamReader(System.in))) {
					String line;
					while ((line = reader.readLine()) != null) {
						String input = handleHereDocument(line, reader);

						float[][] res = processor.apply(input);
						printEmbeddings(out, res, embeddingsFormat);
						out.print("\n> ");
						out.flush();
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
				float[][] res = processor.apply(input);
				printEmbeddings(new PrintWriter(System.out, true, StandardCharsets.UTF_8), res, embeddingsFormat);
			}
		}
	}

	private static void printUsage(PrintStream out) {
		out.println("Usage: java " + JjmlEmbeddings.class.getName() //
				+ " <path/to/model.gguf> [<chunk size>] [ csv | pgvector ]");

		out.println();
		out.println("- Opens a basic interactive chat when in a terminal.");
		out.println("- Piping input will disable interactivity and submit the whole input.");
		out.println("- The context does not auto-extend, that is, it will be full at some point.");
		out.println("- All external inputs should be encoded with UTF-8.");
		out.println("- A <chunk size> of 0 (default) disable chunking.");
		out.println("- Default output format is 'csv', while 'pgvector' generates VALUES.");

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

	/** Computes embeddings based on chunks of a given size. */
	private static class SimpleEmbedding extends LlamaCppEmbeddingProcessor implements Function<String, float[][]> {
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
}
