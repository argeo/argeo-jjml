package tests.mtmd;

import static org.argeo.jjml.llm.LlamaCppContext.defaultContextParams;

import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;

import org.argeo.jjml.imageio.mtmd.ImageIoBitmap;
import org.argeo.jjml.llm.LlamaCppNativeChatFormatter;
import org.argeo.jjml.llm.LlamaCppChatMessage;
import org.argeo.jjml.llm.LlamaCppContext;
import org.argeo.jjml.llm.LlamaCppModel;
import org.argeo.jjml.llm.LlamaCppSamplerChain;
import org.argeo.jjml.llm.LlamaCppSamplers;
import org.argeo.jjml.llm.params.ContextParam;
import org.argeo.jjml.llm.util.InstructRole;
import org.argeo.jjml.mtmd.MtmdBackend;
import org.argeo.jjml.mtmd.MtmdBitmap;
import org.argeo.jjml.mtmd.MtmdContext;
import org.argeo.jjml.mtmd.MtmdImageBitmap;
import org.argeo.jjml.mtmd.MtmdProcessor;

public class DescribeImageTests extends AbstractMtmdTests {

	private String describePrompt = "Describe this image.";

	DescribeImageTests(LlamaCppModel model, Path mmprojPath) {
		super(model, mmprojPath);
	}

	@Override
	protected void all() throws IOException, InterruptedException {
		testDescribeAlpha();
	}

	void testDescribeAlpha() throws IOException {
		Path imagePath = Paths.get("argeo-icon.png");

		try (LlamaCppContext context = new LlamaCppContext(getModel(),
				defaultContextParams().with(ContextParam.n_ctx, 4096)); //
				LlamaCppSamplerChain chain = LlamaCppSamplers.newDefaultSampler(false); //
				MtmdContext mtmdContext = new MtmdContext(getModel(), getMmprojPath(),
						Runtime.getRuntime().availableProcessors()); //
				InputStream imageIn = Files.newInputStream(imagePath); //
		) {
			MtmdProcessor processor = new MtmdProcessor(context, chain, mtmdContext);
			LlamaCppChatMessage systemPrompt = null;
			LlamaCppChatMessage userPrompt = new LlamaCppChatMessage(InstructRole.USER, //
					MtmdBackend.getDefaultMarker() + describePrompt);
			String formatted = new LlamaCppNativeChatFormatter(getModel().getMetadataChatTemplate())
					.formatChatMessages(systemPrompt, userPrompt);
			MtmdImageBitmap bitmap = ImageIoBitmap.load(imageIn);
			MtmdBitmap[] bitmaps = new MtmdBitmap[] { bitmap };

			long begin = System.currentTimeMillis();
			String response = processor.transcribe(formatted, bitmaps);
			System.out.println(response);
			System.out.println("\nProcessing took " + (System.currentTimeMillis() - begin) + " ms");
		}
	}

}
