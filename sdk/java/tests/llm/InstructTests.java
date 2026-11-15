package tests.llm;

import static org.argeo.jjml.llm.LlamaCppContext.defaultContextParams;
import static org.argeo.jjml.llm.LlamaCppModel.defaultModelParams;
import static org.argeo.jjml.llm.params.ContextParam.n_batch;
import static org.argeo.jjml.llm.params.ContextParam.n_ctx;
import static org.argeo.jjml.llm.params.ContextParam.n_threads;
import static org.argeo.jjml.llm.util.InstructRole.SYSTEM;
import static org.argeo.jjml.llm.util.InstructRole.USER;

import java.io.IOException;

import org.argeo.jjml.llm.LlamaCppBackend;
import org.argeo.jjml.llm.LlamaCppContext;
import org.argeo.jjml.llm.LlamaCppInstructProcessor;
import org.argeo.jjml.llm.LlamaCppModel;
import org.argeo.jjml.llm.LlamaCppSamplerChain;
import org.argeo.jjml.llm.LlamaCppSamplers;
import org.argeo.jjml.llm.params.ModelParams;

class InstructTests extends AbstractLlmTests {
	InstructTests(LlamaCppModel model) {
		super(model);
	}

	@Override
	protected void all() throws IOException, InterruptedException {
		testJavaIntroChat();
	}

	void testJavaIntroChat() throws IOException {
		try (//
				LlamaCppContext context = new LlamaCppContext(getModel(), defaultContextParams() //
						.with(n_ctx, 4096) //
						.with(n_batch, 1024) //
						.with(n_threads, getParallelism()) //
				); //
				LlamaCppSamplerChain chain = LlamaCppSamplers.newDefaultSampler(false); //
		) {
			LlamaCppInstructProcessor processor = new LlamaCppInstructProcessor(context, chain);
			processor.setDebugPrompts(System.err);

			String systemMsg = "You are a helpful assistant, which answers as briefly as possible.";
//			logger.log(DEBUG, SYSTEM.name() + " :\n" + systemMsg);
			processor.write(SYSTEM, systemMsg);

			String userMsg01 = "Introduce the Java programming language in no more than two sentences.";
//			logger.log(DEBUG, USER.name() + " :\n" + userMsg01);
			processor.write(USER, userMsg01);

//			processor.write(ASSISTANT, "");
//			logger.log(DEBUG, ASSISTANT.name() + " :\n");
			processor.readMessage(out);

			// make sure it can deal with a second message
			String userMsg02 = "Thank you!";
//			logger.log(DEBUG, USER.name() + " :\n" + userMsg02);
			processor.write(USER, userMsg02);

//			processor.write(ASSISTANT, "");
//			logger.log(DEBUG, ASSISTANT.name() + " :\n");
			processor.readMessage(out);
		}
	}

	public static void main(String[] args) throws Exception {
		if (args.length < 1)
			throw new IllegalArgumentException("Specify a model");
		String modelHint = args[0];
		ModelParams modelParams = defaultModelParams();
		try (LlamaCppModel model = AbstractLlmTests.createModel(modelParams, modelHint)) {
			new InstructTests(model).run();
		} finally {
			LlamaCppBackend.destroy();
		}
	}

}
