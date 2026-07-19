//!/usr/bin/env -S java -ea -cp /usr/share/java/org.argeo.jjml.jar
package examples;

import static org.argeo.jjml.llm.LlamaCppContext.defaultContextParams;
import static org.argeo.jjml.llm.LlamaCppModel.defaultModelParams;
import static org.argeo.jjml.llm.params.ContextParam.n_batch;
import static org.argeo.jjml.llm.params.ContextParam.n_ctx;
import static org.argeo.jjml.llm.params.ContextParam.n_threads;
import static org.argeo.jjml.llm.util.InstructRole.SYSTEM;
import static org.argeo.jjml.llm.util.InstructRole.USER;

import java.io.InputStream;
import java.io.StringWriter;
import java.net.URI;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.StandardCopyOption;
import java.util.concurrent.Future;

import org.argeo.jjml.llm.LlamaCppBackend;
import org.argeo.jjml.llm.LlamaCppContext;
import org.argeo.jjml.llm.LlamaCppInstructProcessor;
import org.argeo.jjml.llm.LlamaCppModel;
import org.argeo.jjml.llm.LlamaCppNative;
import org.argeo.jjml.llm.LlamaCppSamplerChain;
import org.argeo.jjml.llm.LlamaCppSamplers;
import org.argeo.jjml.llm.params.ModelParams;

/**
 * Minimal set of non-destructive in-memory tests, in order to check that a
 * given deployment and/or model are working. Java assertions must be enabled.
 */
class JjmlSmokeTests {
	private static boolean allPassed = true;

	public static void main(String[] args) {
		try {
			LlamaCppNative.ensureLibrariesLoaded();
			System.out.println("Native libraries properly loaded.");

			final Path modelPath;
			if (args.length == 0) {
				Path downloadedModel = Paths.get(System.getProperty("java.io.tmpdir"), "jjml-smoke-tests-model.gguf");
				if (Files.exists(downloadedModel)) {
					System.out.println("No model was specified, using " + downloadedModel);
				} else {
					String url = "https://huggingface.co/ibm-granite/granite-4.0-350m-GGUF/resolve/main/granite-4.0-350m-Q4_0.gguf";
					System.out.println("No model was specified, downloading a tiny model from " + url + " ...");
					try (InputStream in = URI.create(url).toURL().openStream()) {
						Files.copy(in, downloadedModel, StandardCopyOption.REPLACE_EXISTING);
					}
				}
				modelPath = downloadedModel;
			} else {
				modelPath = Paths.get(args[0]);
				if (!Files.exists(modelPath))
					throw new IllegalArgumentException("Could not find GGUF model " + modelPath);
			}

			ModelParams modelParams = defaultModelParams();
			System.out.println("Loading model " + modelPath.getFileName() + " ...");
			Future<LlamaCppModel> loaded = LlamaCppModel.loadAsync(modelPath, modelParams, null, null);
			try (//
					LlamaCppModel model = loaded.get(); //
					LlamaCppContext context = new LlamaCppContext(model, defaultContextParams() //
							.with(n_ctx, 1024) //
							.with(n_batch, 512) //
							.with(n_threads, Runtime.getRuntime().availableProcessors()) //
					); //
					LlamaCppSamplerChain chain = LlamaCppSamplers.newDefaultSampler(false); //
			) {
				LlamaCppInstructProcessor processor = new LlamaCppInstructProcessor(context, chain);
				processor.setDebugPrompts(System.out);

				String systemMsg = "You are a helpful assistant.";
//				System.out.println(SYSTEM.name() + " :\n" + systemMsg);
				processor.write(SYSTEM, systemMsg);

				String userMsg01 = "In one word, the capital of France is ";
//				System.out.println(USER.name() + " :\n" + userMsg01);
				processor.write(USER, userMsg01);

//				System.out.println(ASSISTANT.name() + " :\n");
				StringWriter answerSW = new StringWriter();
				processor.readMessage(answerSW);

				String answer = answerSW.toString();
				System.out.println(answer);

				String sanitized = answer.strip().replaceAll("[^a-zA-Z]", "");
				if (!"paris".equals(sanitized.toLowerCase()))
					allPassed = false;
			}
		} catch (Exception | AssertionError e) {
			allPassed = false;
			System.err.println("Smoke tests failed");
			e.printStackTrace();
		} finally {
			LlamaCppBackend.destroy();
		}

		if (allPassed)
			System.exit(0);
		else
			System.exit(1);
	}
}
