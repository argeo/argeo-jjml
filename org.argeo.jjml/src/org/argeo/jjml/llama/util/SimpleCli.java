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
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;

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
		String systemPrompt = "You are a helpful assistant.";
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

		try (LlamaCppModel model = LlamaCppModel.load(modelPath, defaultModelParams()); //
				LlamaCppContext context = new LlamaCppContext(model, defaultContextParams()); //
		) {
			SimpleChat chat = new SimpleChat(systemPrompt, context);

			Console console = System.console();
			final boolean isConsoleTerminal = console != null;
			// From Java 22, it will be:
			// boolean interactive = console.isTerminal();
			final boolean developing = false; // force true in IDE while developing
			final boolean interactive = developing || isConsoleTerminal;

			try {
				if (interactive) {
					System.out.print("> ");
					try (BufferedReader reader = new BufferedReader(
							console != null ? console.reader() : new InputStreamReader(System.in))) {
						String line;
						while ((line = reader.readLine()) != null) {
							String input = handleHereDocument(line, reader);
							chat.apply(input, (str) -> {
								System.out.print(str);
								System.out.flush();
							}).thenAccept((v) -> {
								System.out.print("\n> ");
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
//					String input = in.lines().collect(Collectors.joining("\n"));
					chat.apply(input, (str) -> {
						System.out.print(str);
						System.out.flush();
					}).toCompletableFuture().join();
				}
			} finally {
				// make sure that we are not reading before cleaning up backend
				chat.cancelCurrentRead();
			}
		}
	}

	private static void printUsage(PrintStream out) {
		out.println("Usage: java " + SimpleCli.class.getName() //
				+ " <path/to/model.gguf> [<system prompt>]");

		out.println();
		out.println("- In a terminal, this will open an interactive chat.");
		out.println("- Piping input will disable interactivity and submit the whole input as a single user prompt.");
		out.println("- The context does not auto-extend, that is, it will be full at some point.");
		out.println("- All external inputs should be encoded with UTF-8.");
		out.println("- If <system prompt> contains a file separator or /, it will be loaded as a file.");

		out.println();
		out.println("In interactive mode, use <<EOF for multi-line input. For example:");
		out.println();
		out.println("> Suggest improvements to this Java code: <<EOF");
		out.println("public static void main(String[] args) {");
		out.println("  System.out.println(\"Hello world!\");");
		out.println("}");
		out.println("EOF");

		out.println();
		out.println("# System properties for supported parameters (see llama.h for details):");
		for (ModelParam param : ModelParam.values())
			out.println("-D" + ModelParam.SYSTEM_PROPERTY_MODEL_PARAM_PREFIX + param + "=");
		for (ContextParam param : ContextParam.values())
			out.println("-D" + ContextParam.SYSTEM_PROPERTY_CONTEXT_PARAM_PREFIX + param + "=");

		out.println();
		out.println("# System properties for explicit paths to shared libraries:");
		out.println("-D" + LlamaCppNative.SYSTEM_PROPERTY_LIBPATH_JJML_LLAMA + "=");
		out.println("-D" + LlamaCppNative.SYSTEM_PROPERTY_LIBPATH_LLAMACPP + "=");
		out.println("-D" + LlamaCppNative.SYSTEM_PROPERTY_LIBPATH_GGML + "=");
	}

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
