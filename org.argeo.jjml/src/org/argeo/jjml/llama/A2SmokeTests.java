package org.argeo.jjml.llama;

import static java.lang.System.Logger.Level.ERROR;
import static java.lang.System.Logger.Level.INFO;
import static java.lang.System.Logger.Level.DEBUG;
import static java.nio.charset.StandardCharsets.UTF_8;
import static org.argeo.jjml.llama.LlamaCppContext.defaultContextParams;
import static org.argeo.jjml.llama.LlamaCppModel.defaultModelParams;
import static org.argeo.jjml.llama.LlamaCppSamplers.newJavaSampler;
import static org.argeo.jjml.llama.params.ContextParam.embeddings;
import static org.argeo.jjml.llama.params.ContextParam.n_batch;
import static org.argeo.jjml.llama.params.ContextParam.n_ctx;
import static org.argeo.jjml.llama.params.ContextParam.n_ubatch;
import static org.argeo.jjml.llama.util.StandardRole.SYSTEM;
import static org.argeo.jjml.llama.util.StandardRole.USER;

import java.io.IOException;
import java.io.UncheckedIOException;
import java.lang.System.Logger;
import java.lang.System.Logger.Level;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.IntBuffer;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.time.LocalDateTime;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Set;
import java.util.concurrent.Future;
import java.util.function.BooleanSupplier;
import java.util.function.Consumer;
import java.util.function.DoubleConsumer;

import org.argeo.jjml.llama.params.ContextParams;
import org.argeo.jjml.llama.params.ModelParams;

/**
 * Minimal set of non-destructive in-memory tests, in order to check that a
 * given deployment and/or model are working. Java assertions must be enabled.
 */
class A2SmokeTests {
	private final static Logger logger = System.getLogger(A2SmokeTests.class.getName());

	public void main(List<String> args) throws Exception, AssertionError {
		try {
			if (!getClass().desiredAssertionStatus()) {
				logger.log(ERROR, "Assertions must be enabled. Please call Java with the -ea option.");
				return;
			}
			if (args.isEmpty()) {
				logger.log(ERROR, "Usage: " + getClass().getSimpleName() + " <path to GGUF model>");
				return;
			}
			Path modelPath = Paths.get(args.get(0));

			assert ((BooleanSupplier) () -> {
				LlamaCppNative.ensureLibrariesLoaded();
				return true;
			}).getAsBoolean();

			ModelParams modelParams = defaultModelParams();
			logger.log(INFO, "Loading model " + modelPath + " ...");
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
				logger.log(DEBUG, "Metadata:\n" + sb);

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
				assertJavaSampler(model);
				assertChat(model);
				assertSavedContextState(model);
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
		logger.log(INFO, "Vocabulary smoke tests variant PASSED");
	}

	boolean testTokenizeDetokenize(LlamaCppVocabulary vocabulary, ByteBuffer in, IntBuffer buf, String msg) {
		if (in != null)
			in.clear();
		buf.clear();

		logger.log(DEBUG, msg);
		if (in == null) {
			IntBuffer tokens = vocabulary.tokenize(msg);
			buf.put(tokens);
		} else {
			in.put(msg.getBytes(UTF_8));
			in.flip();
			vocabulary.tokenize(msg, buf);
		}
		buf.flip();
		logger.log(DEBUG, LlamaCppVocabulary.logIntegers(buf, 32, ", "));
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
			LlamaCppEmbeddingProcessor embeddingProcessor = new LlamaCppEmbeddingProcessor(context);

			List<String> prompts = new ArrayList<>();
			prompts.add("Hello world!");
			prompts.add("Good night and good luck.");
			for (String s : prompts)
				logger.log(DEBUG, "=>\n" + s);

			float[][] embeddings = embeddingProcessor.processEmbeddings(prompts);
			assert embeddings.length != 0;

			for (float[] embedding : embeddings) {
				logger.log(DEBUG, "<=\n[ " + embedding[0] + ", " + embedding[1] + ", ... ]");
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
			LlamaCppTextProcessor processor = new LlamaCppTextProcessor(context, chain, validatingSampler,
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

	void assertJavaSampler(LlamaCppModel model) {
		Integer[] sequenceIds = { 1 };
		try ( //
				LlamaCppContext context = new LlamaCppContext(model, defaultContextParams() //
						.with(n_ctx, 6144) //
						.with(n_batch, sequenceIds.length * 64)); //
				LlamaCppSamplerChain chain = new LlamaCppSamplerChain(
						newJavaSampler(new LlamaCppJavaSampler.SimpleGreedy())); //
				LlamaCppNativeSampler validatingSampler = LlamaCppSamplers.newSamplerGrammar(model, //
						"root ::= [ \\t\\n]* \"TEST\"", "root");//
		) {
//			long begin = System.currentTimeMillis();
			LlamaCppTextProcessor processor = new LlamaCppTextProcessor(context, chain, validatingSampler,
					Set.of(sequenceIds));

			String prompt = "Write HELLO\n"//
					+ "Hello\n"//
					+ "Write World\n"//
					+ "WORLD\n"//
					+ "Write test\n" //
			;
			logger.log(INFO, "=>\n" + prompt);
			String str = processor.processBatch(prompt);
			logger.log(INFO, "<=\n" + str);
			// System.out.println("\n\n## Processing took " + (System.currentTimeMillis() -
			// begin) + " ms");

		}
		logger.log(INFO, "Java sampler smoke tests PASSED");
	}

	void assertChat(LlamaCppModel model) throws IOException {
		try (//
				LlamaCppContext context = new LlamaCppContext(model, defaultContextParams() //
						.with(n_ctx, 20480) //
						.with(n_batch, 1024)); //
				LlamaCppSamplerChain chain = LlamaCppSamplers.newDefaultSampler(model, false); //
		) {
			LlamaCppInstructProcessor processor = new LlamaCppInstructProcessor(context, chain);

			String systemMsg = "You are a helpful assistant, which answer as briefly as possible.";
			logger.log(INFO, SYSTEM.name() + " : " + systemMsg);
			processor.write(SYSTEM, systemMsg);

			String userMsg01 = "Introduce the Java programming language in no more than two sentences.";
			logger.log(INFO, USER.name() + " : " + userMsg01);
			processor.write(USER, userMsg01);

			processor.readMessage(System.out);

			// make sure it can deal with a second message
			String userMsg02 = "Thank you!";
			logger.log(INFO, USER.name() + " : " + userMsg02);
			processor.write(USER, userMsg02);

			processor.readMessage(System.out);
		}
		logger.log(INFO, "Chat smoke tests PASSED");
	}

	void assertSavedContextState(LlamaCppModel model) throws IOException {
		ContextParams contextParams = LlamaCppContext.defaultContextParams() //
				.with(n_ctx, 20480) //
				.with(n_batch, 1024) //
		; //

		final LlamaCppContextState savedState;
		try (//
				LlamaCppContext context = new LlamaCppContext(model, contextParams); //
				LlamaCppSamplerChain chain = LlamaCppSamplers.newDefaultSampler(model, false); //
		) {
			LlamaCppInstructProcessor processor = new LlamaCppInstructProcessor(context, chain);

			long begin = System.currentTimeMillis();
			String systemMsg = "You are a travel agent helping the user to chose the best holiday destination.\n"
					+ "You answer with a city name, and one sentence explanation of your choice, nothing else.";
			logger.log(INFO, SYSTEM.name() + " : " + systemMsg);
			processor.write(SYSTEM, systemMsg);

			String userMsg01 = "I want to spend my vacations in Europe.\n"
					+ "I like Italy, but I am open to other destinations, as long as there is nature and culture.\n"
					+ "I have never been to Scandinavia, but it can wait.\n"
					+ "I would like to avoid the usual touristic destinations, so be creative!\n"
					+ "I will travel in autumn, so it should not be too hot.\n"
					+ "Also please consider that I speak French and German in addition to English.\n"
					+ "And I definitely don't like holiday on the beach...";
			logger.log(INFO, USER.name() + " : " + userMsg01);
			processor.write(USER, userMsg01);

			savedState = new LlamaCppContextState.ByteBufferSavedState();
			processor.saveContextState(savedState);
			logger.log(INFO, "Wrote and saved context in " + (System.currentTimeMillis() - begin) + " ms");
		}

		String userMsg02 = "Current Date: " + LocalDateTime.now();

		Consumer<LlamaCppInstructProcessor> process = (processor) -> {
			long beginLoad = System.currentTimeMillis();
			processor.loadContextState(savedState);
			logger.log(INFO, "Loaded context in " + (System.currentTimeMillis() - beginLoad) + " ms");

			logger.log(INFO, USER.name() + " : " + userMsg02);
			processor.write(USER, userMsg02);

			long begin = System.currentTimeMillis();
			try {
				processor.readMessage(System.out);
			} catch (IOException e) {
				throw new UncheckedIOException(e);
			}
			logger.log(INFO, "Generation took " + +(System.currentTimeMillis() - begin) + " ms\n\n");
		};

		// deterministic answer
		try (LlamaCppContext context = new LlamaCppContext(model, contextParams); //
				LlamaCppSamplerChain chain = LlamaCppSamplers.newDefaultSampler(model, false); //
		) {
			process.accept(new LlamaCppInstructProcessor(context, chain));
		}

		// with temperature
		try (LlamaCppContext context = new LlamaCppContext(model, contextParams); //
				LlamaCppSamplerChain chain = LlamaCppSamplers.newDefaultSampler(model, true); //
		) {
			process.accept(new LlamaCppInstructProcessor(context, chain));
		}

		logger.log(INFO, "Saved context state smoke tests PASSED");
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
