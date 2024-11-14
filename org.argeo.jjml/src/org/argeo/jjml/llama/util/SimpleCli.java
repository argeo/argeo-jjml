package org.argeo.jjml.llama.util;

import static java.nio.charset.StandardCharsets.UTF_8;
import static org.argeo.jjml.llama.LlamaCppContext.defaultContextParams;
import static org.argeo.jjml.llama.LlamaCppModel.defaultModelParams;

import java.io.BufferedReader;
import java.io.Console;
import java.io.File;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.PrintStream;
import java.io.PrintWriter;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.StringJoiner;
import java.util.function.Supplier;

import org.argeo.jjml.llama.LlamaCppContext;
import org.argeo.jjml.llama.LlamaCppModel;
import org.argeo.jjml.llama.LlamaCppNative;
import org.argeo.jjml.llama.params.ContextParam;
import org.argeo.jjml.llama.params.ModelParam;

/** A minimal command line interface for batch processing and simple chat. */
public class SimpleCli {
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
		Path modelPath = Paths.get(args[0]);

		boolean embeddings = Boolean.parseBoolean(
				System.getProperty(ContextParam.SYSTEM_PROPERTY_CONTEXT_PARAM_PREFIX + ContextParam.embeddings));
		int chunkSize = 0;
		String embeddingsFormat = "csv";
		String systemPrompt = "You are a helpful assistant.";
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

		try (LlamaCppModel model = LlamaCppModel.load(modelPath, defaultModelParams()); //
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
			final boolean developing = false; // force true in IDE while developing
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
		out.println("- For embeddings, a <chunk size> of 0 disable chunking.");
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
			out.println("-D" + ModelParam.SYSTEM_PROPERTY_MODEL_PARAM_PREFIX + param + "=");
		for (ContextParam param : ContextParam.values())
			out.println("-D" + ContextParam.SYSTEM_PROPERTY_CONTEXT_PARAM_PREFIX + param + "=");

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
