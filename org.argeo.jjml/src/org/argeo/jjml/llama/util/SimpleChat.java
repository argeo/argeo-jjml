package org.argeo.jjml.llama.util;

import static org.argeo.jjml.llama.util.StandardRole.SYSTEM;

import java.nio.IntBuffer;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.CompletionStage;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ForkJoinPool;
import java.util.concurrent.FutureTask;
import java.util.function.BiFunction;
import java.util.function.Consumer;

import org.argeo.jjml.llama.LlamaCppBatchProcessor;
import org.argeo.jjml.llama.LlamaCppChatMessage;
import org.argeo.jjml.llama.LlamaCppContext;
import org.argeo.jjml.llama.LlamaCppSamplerChain;
import org.argeo.jjml.llama.LlamaCppSamplers;
import org.argeo.jjml.llama.LlamaCppVocabulary;

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
		this(systemPrompt, context, LlamaCppSamplers.newDefaultSampler(context.getModel(), true));
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
			systemMsg = SYSTEM.msg(systemPrompt);
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
			LlamaCppChatMessage userMsg = StandardRole.USER.msg(message);
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
}
