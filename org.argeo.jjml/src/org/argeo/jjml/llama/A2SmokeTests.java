package org.argeo.jjml.llama;

import static java.lang.System.Logger.Level.INFO;
import static java.nio.charset.StandardCharsets.UTF_8;
import static org.argeo.jjml.llama.LlamaCppChatMessage.StandardRole.SYSTEM;
import static org.argeo.jjml.llama.LlamaCppChatMessage.StandardRole.USER;
import static org.argeo.jjml.llama.LlamaCppContext.defaultContextParams;
import static org.argeo.jjml.llama.LlamaCppModel.defaultModelParams;
import static org.argeo.jjml.llama.params.ContextParamName.embeddings;
import static org.argeo.jjml.llama.params.ContextParamName.n_batch;
import static org.argeo.jjml.llama.params.ContextParamName.n_ctx;
import static org.argeo.jjml.llama.params.ContextParamName.n_ubatch;

import java.lang.System.Logger;
import java.lang.System.Logger.Level;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import java.nio.IntBuffer;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Set;
import java.util.concurrent.Future;
import java.util.function.BooleanSupplier;
import java.util.function.DoubleConsumer;

import org.argeo.jjml.llama.params.ModelParams;

/**
 * Minimal set of non-destructive in-memory tests, in order to check that a
 * given deployment and/or model are working. Java assertions must be enabled.
 */
class A2SmokeTests {
	private final static Logger logger = System.getLogger(A2SmokeTests.class.getName());

	static {
		assert ((BooleanSupplier) () -> {
			LlamaCppNative.ensureLibrariesLoaded();
			LlamaCppBackend.ensureInitialized();
			return true;
		}).getAsBoolean();
	}

	public void main(List<String> args) throws Exception, AssertionError {
		try {
			if (args.isEmpty())
				return;
			Path modelPath = Paths.get(args.get(0));

			ModelParams modelParams = defaultModelParams();
			Future<LlamaCppModel> loaded = LlamaCppModel.loadAsync(modelPath, modelParams,
					new LoadModelProgressCallback(), null);
			try (LlamaCppModel model = loaded.get();) {
				logger.log(INFO, "Model " + model.getDescription());
				logger.log(INFO, model.getLayerCount() + " layers");
				logger.log(INFO, model.getEmbeddingSize() + " embedding size");
				logger.log(INFO, model.getVocabularySize() + " vocabulary size");
				logger.log(INFO, model.getContextTrainingSize() + " context training size");
				StringBuilder sb = new StringBuilder();
				for (String key : model.getMetadata().keySet())
					sb.append(key + "=" + model.getMetadata().get(key) + "\n");
				logger.log(INFO, "Metadata:\n" + sb);

				model.getVocabulary().setStringMode(false);
				assertVocabulary(model.getVocabulary());
				model.getVocabulary().setStringMode(true);
				assertVocabulary(model.getVocabulary());
				// TODO return if vocabulary only
//				if (true)
//					return;

				model.getVocabulary().setStringMode(false);
				assertLoadUnloadDefaultContext(model);
				assertEmbeddings(model);
				assertBatch(model);
				assertChat(model);
			}
		} catch (Exception | AssertionError e) {
			logger.log(Level.ERROR, "Smoke tests failed", e);
			throw e;
		} finally {
			LlamaCppBackend.destroy();
		}
	}

	void assertVocabulary(LlamaCppVocabulary vocabulary) {
		int size = 256;

		// in direct, out direct
		assertVocabulary(vocabulary, //
				ByteBuffer.allocateDirect(size), //
				ByteBuffer.allocateDirect(size).order(ByteOrder.nativeOrder()).asIntBuffer());
		// in array, out direct
		assertVocabulary(vocabulary, //
				ByteBuffer.allocate(size), //
				ByteBuffer.allocateDirect(size).order(ByteOrder.nativeOrder()).asIntBuffer());
		// in string, out direct
		assertVocabulary(vocabulary, //
				null, //
				ByteBuffer.allocateDirect(size).order(ByteOrder.nativeOrder()).asIntBuffer());
		// in direct, out array
		assertVocabulary(vocabulary, //
				ByteBuffer.allocateDirect(size), //
				IntBuffer.allocate(size / Integer.BYTES));
		// in array, out array
		assertVocabulary(vocabulary, //
				ByteBuffer.allocate(size), //
				IntBuffer.allocate(size / Integer.BYTES));
		// in string, out array
		assertVocabulary(vocabulary, //
				null, //
				IntBuffer.allocate(size / Integer.BYTES));
	}

	void assertVocabulary(LlamaCppVocabulary vocabulary, ByteBuffer in, IntBuffer out) {
		assert testTokenizeDetokenize(vocabulary, in, out, "Hello World!");
		assert testTokenizeDetokenize(vocabulary, in, out, "Même si je suis Français, je dis bonjour au monde");
		assert testTokenizeDetokenize(vocabulary, in, out, "ἔορθoι χθόνιοι"); // according to olmoe-1b-7b-0924
		assert testTokenizeDetokenize(vocabulary, in, out, "السلام عليكم"); // according to olmoe-1b-7b-0924
		assert testTokenizeDetokenize(vocabulary, in, out, "¡Hola и أَشْكَرُ мир! 👋🏼🌍");
		logger.log(INFO, "Vocabulary smoke tests PASSED");
	}

	boolean testTokenizeDetokenize(LlamaCppVocabulary vocabulary, ByteBuffer in, IntBuffer buf, String msg) {
		if (in != null)
			in.clear();
		buf.clear();

		logger.log(INFO, msg);
		if (in == null) {
			IntBuffer tokens = vocabulary.tokenize(msg);
			buf.put(tokens);
		} else {
			in.put(msg.getBytes(UTF_8));
			in.flip();
			vocabulary.tokenize(msg, buf);
		}
		buf.flip();
		logger.log(INFO, LlamaCppVocabulary.logIntegers(buf, 32, ", "));
		String str;
		if (in == null) {
			str = vocabulary.deTokenize(buf);
		} else {
			in.clear();
			vocabulary.deTokenize(buf, in);
			in.flip();
			str = UTF_8.decode(in).toString();
		}
		assert str.equals(msg);
		return true;
	}

	void assertLoadUnloadDefaultContext(LlamaCppModel model) {
		try (LlamaCppContext context = new LlamaCppContext(model);) {
			assert context.getContextSize() > 0;
		}
		logger.log(INFO, "Load default context smoke tests PASSED");
	}

	void assertEmbeddings(LlamaCppModel model) {
		int batchSize = 512;
		try (LlamaCppContext context = new LlamaCppContext(model, LlamaCppContext.defaultContextParams() //
				.with(embeddings, true) //
				.with(n_ctx, 6144) //
				.with(n_batch, batchSize) //
				.with(n_ubatch, batchSize) // must be same for embeddings
		);) {
			LlamaCppEmbeddingProcessor embeddingProcessor = new LlamaCppEmbeddingProcessor();
			embeddingProcessor.setContext(context);

			List<String> prompts = new ArrayList<>();
			prompts.add("Hello world!");
			prompts.add("Good night and good luck.");
			for (String s : prompts)
				logger.log(INFO, "=>\n" + s);

			// long begin = System.currentTimeMillis();
			List<FloatBuffer> embeddings = embeddingProcessor.processEmbeddings(prompts);
			assert !embeddings.isEmpty();
			// System.out.println("\n\n## Processing took " + (System.currentTimeMillis() -
			// begin) + " ms");

			for (FloatBuffer embedding : embeddings) {
				logger.log(INFO, "<=\n" + embedding);
			}
		}
		logger.log(INFO, "Embeddings smoke tests PASSED");
	}

	void assertBatch(LlamaCppModel model) {
		Integer[] sequenceIds = { 1, 10, 100 };
		try ( //
				LlamaCppContext context = new LlamaCppContext(model, defaultContextParams() //
						.with(n_ctx, 6144) //
						.with(n_batch, sequenceIds.length * 64)); //
				LlamaCppSamplerChain chain = LlamaCppSamplers.newDefaultSampler(model, false); //
				LlamaCppNativeSampler validatingSampler = LlamaCppSamplers.newSamplerGrammar(model, //
						"root ::= [ \\t\\n]* \"TEST\"", "root");//
		) {
//			long begin = System.currentTimeMillis();
			LlamaCppBatchProcessor processor = new LlamaCppBatchProcessor(context, chain, validatingSampler,
					Set.of(sequenceIds));

			String prompt = "Write HELLO\n"//
					+ "HELLO\n"//
					+ "Write WORLD\n"//
					+ "WORLD\n"//
					+ "Write TEST\n" //
			;
			logger.log(INFO, "=>\n" + prompt);
			String str = processor.processBatch(prompt);
			logger.log(INFO, "<=\n" + str);
			// System.out.println("\n\n## Processing took " + (System.currentTimeMillis() -
			// begin) + " ms");

		}
		logger.log(INFO, "Batch smoke tests PASSED");
	}

	void assertChat(LlamaCppModel model) {
		try (//
				LlamaCppContext context = new LlamaCppContext(model, defaultContextParams() //
						.with(n_ctx, 20480) //
						.with(n_batch, 1024)); //
				LlamaCppSamplerChain chain = LlamaCppSamplers.newDefaultSampler(model, true); //
		) {
			LlamaCppBatchProcessor processor = new LlamaCppBatchProcessor(context, chain);

//			List<LlamaCppChatMessage> messages = new ArrayList<>();
//			String previousPrompts = "";
//			messages.add(SYSTEM.msg("You are a helpful assistant"));

			String prompt;
			String reply;

			prompt = model.formatChatMessages( //
					SYSTEM.msg("You are a helpful assistant."), //
					USER.msg("Briefly introduce the Java programming language."));
			reply = processor.processSingleBatch(prompt);
			logger.log(INFO, "\n" + prompt + reply);

			prompt = model.formatChatMessages(USER.msg("Thank you!"));
			reply = processor.processSingleBatch(prompt);
			logger.log(INFO, "\n" + prompt + reply);

//			messages.add(USER.msg("Briefly introduce the Java programming language."));
//			logger.log(INFO, "=>\n" + messages.get(messages.size() - 1).getContent());
//			prompt = ((BiFunction<String, String, String>) (p, s) //
//			-> s.substring(p.length(), s.length()) //
//			).apply(previousPrompts, model.formatChatMessages(messages));
//			reply = processor.processSingleBatch(prompt, 0);
//			messages.add(ASSISTANT.msg(reply));
//			logger.log(INFO, "<=\n" + messages.get(messages.size() - 1).getContent());
//			previousPrompts = model.formatChatMessages(messages);
//
//			messages.add(USER.msg("Thank you!"));
//			logger.log(INFO, "=>\n" + messages.get(messages.size() - 1).getContent());
//			prompt = ((BiFunction<String, String, String>) (p, s) //
//			-> s.substring(p.length(), s.length()) //
//			).apply(previousPrompts, model.formatChatMessages(messages));
//			reply = processor.processSingleBatch(prompt, 0);
//			messages.add(ASSISTANT.msg(reply));
//			logger.log(INFO, "<=\n" + messages.get(messages.size() - 1).getContent());
//			previousPrompts = model.formatChatMessages(messages);
		}
		logger.log(INFO, "Chat smoke tests PASSED");
	}

	/*
	 * UTILITIES
	 */

	public static void main(String[] args) throws Exception {
		new A2SmokeTests().main(Arrays.asList(args));
	}

	/*
	 * CLASSES
	 */
	static class LoadModelProgressCallback implements DoubleConsumer {
		private int lastPerctPrinted = -1;

		@Override
		public void accept(double progress) {
			char[] progressBar = new char[10];
			int perct = (int) (progress * 100);

			if (perct > lastPerctPrinted + 10 //
					|| lastPerctPrinted == -1 //
					|| progress == 1.0) {

				for (int i = 0; i < perct / 10; i++)
					progressBar[i] = '#';
				for (int i = perct / 10; i < 10; i++)
					progressBar[i] = '-';
				System.err.print("\r" + new String(progressBar));

				lastPerctPrinted = perct;
				if (progress == 1.0)
					System.out.print("\n");
			}
		}

	}
}
