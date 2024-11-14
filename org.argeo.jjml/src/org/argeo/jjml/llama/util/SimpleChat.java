package org.argeo.jjml.llama.util;

import static org.argeo.jjml.llama.LlamaCppContext.defaultContextParams;
import static org.argeo.jjml.llama.LlamaCppModel.defaultModelParams;
import static org.argeo.jjml.llama.util.StandardRole.SYSTEM;

import java.io.BufferedReader;
import java.io.Console;
import java.io.InputStreamReader;
import java.io.PrintStream;
import java.nio.IntBuffer;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.CompletionStage;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ForkJoinPool;
import java.util.concurrent.FutureTask;
import java.util.function.BiFunction;
import java.util.function.Consumer;
import java.util.stream.Collectors;

import org.argeo.jjml.llama.LlamaCppBatchProcessor;
import org.argeo.jjml.llama.LlamaCppChatMessage;
import org.argeo.jjml.llama.LlamaCppContext;
import org.argeo.jjml.llama.LlamaCppModel;
import org.argeo.jjml.llama.LlamaCppNative;
import org.argeo.jjml.llama.LlamaCppSamplerChain;
import org.argeo.jjml.llama.LlamaCppSamplers;
import org.argeo.jjml.llama.LlamaCppVocabulary;
import org.argeo.jjml.llama.params.ContextParam;
import org.argeo.jjml.llama.params.ModelParam;

/**
 * A simple implementation of a chat system based on the low-level components.
 */
public class SimpleChat extends LlamaCppBatchProcessor
		implements BiFunction<String, Consumer<String>, CompletionStage<Void>> {
	private final LlamaCppVocabulary vocabulary;

	private volatile boolean reading = false;
	private FutureTask<Void> currentRead = null;

	private final LlamaCppChatMessage systemMsg;
	private boolean firstMessage = true;

	/*
	 * In llama.cpp examples/main, a new user prompt is obtained via a substring of
	 * the previous messages. We reproduce this behavior here, even though it is not
	 * clear yet whether it is useful.
	 */
	// TODO: check in details and remove this as it would greatly simplify the code
	private final boolean usePreviousMessages;
	private final List<LlamaCppChatMessage> messages;

	public SimpleChat(String systemPrompt, LlamaCppContext context) {
		this(systemPrompt, context, LlamaCppSamplers.newDefaultSampler(context.getModel(), true));
	}

	public SimpleChat(String systemPrompt, LlamaCppContext context, LlamaCppSamplerChain samplerChain) {
		super(context, samplerChain);
		vocabulary = getModel().getVocabulary();
		systemMsg = SYSTEM.msg(systemPrompt);

		usePreviousMessages = true;
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
		LlamaCppChatMessage userMsg = StandardRole.USER.msg(message);
		String prompt;
		if (usePreviousMessages) {
			String previousPrompts = messages.size() == 0 ? "" : getModel().formatChatMessages(messages);
			if (firstMessage) {
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

		// tokenize
		IntBuffer input = vocabulary.tokenize(prompt);
		writeBatch(input, true);
		FutureTask<Void> future = new FutureTask<>(() -> {
			String reply = readAll(consumer);
			if (usePreviousMessages) {
				LlamaCppChatMessage assistantMsg = StandardRole.ASSISTANT.msg(reply);
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
		if (args.length > 1)
			systemPrompt = args[1];

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

			CompletionStage<Void> reply = CompletableFuture.completedStage(null);
			try (BufferedReader in = new BufferedReader(
					console != null ? console.reader() : new InputStreamReader(System.in));) {
				if (interactive) {
					System.out.print("> ");
					String line;
					while ((line = in.readLine()) != null) {
						// start chat cycle
						reply = chat.apply(line, (str) -> {
							System.out.print(str);
							System.out.flush();
						}).thenAccept((v) -> {
							System.out.print("\n> ");
						});
					}
				} else {// batch
					String input = in.lines().collect(Collectors.joining("\n"));
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
		out.println("Usage: java " + SimpleChat.class.getName() //
				+ " <path/to/model.gguf> [<system prompt>]");

		out.println();
		out.println("In a terminal, it will open an interactive chat.");
		out.println("Piping input will disable interactivity and the additional characters in the output.");
		out.println("The context does not auto-extend, that is, it will be full at some point.");

		out.println();
		out.println("System properties for supported parameters (see llama.h for details):");
		for (ModelParam param : ModelParam.values())
			out.println("-D" + ModelParam.SYSTEM_PROPERTY_MODEL_PARAM_PREFIX + param + "=");
		for (ContextParam param : ContextParam.values())
			out.println("-D" + ContextParam.SYSTEM_PROPERTY_CONTEXT_PARAM_PREFIX + param + "=");

		out.println();
		out.println("System properties for explicit paths to shared libraries:");
		out.println("-D" + LlamaCppNative.SYSTEM_PROPERTY_LIBPATH_JJML_LLAMA + "=");
		out.println("-D" + LlamaCppNative.SYSTEM_PROPERTY_LIBPATH_LLAMACPP + "=");
		out.println("-D" + LlamaCppNative.SYSTEM_PROPERTY_LIBPATH_GGML + "=");
	}
}
