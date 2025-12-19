package org.argeo.jjml.imageio.mtmd;

import static org.argeo.jjml.llm.LlamaCppContext.defaultContextParams;

import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;

import org.argeo.jjml.llm.LlamaCppContext;
import org.argeo.jjml.llm.LlamaCppModel;
import org.argeo.jjml.llm.LlamaCppSamplerChain;
import org.argeo.jjml.llm.LlamaCppSamplers;
import org.argeo.jjml.llm.params.ContextParam;
import org.argeo.jjml.llm.util.SimpleModelDownload;
import org.argeo.jjml.mtmd.MtmdBackend;
import org.argeo.jjml.mtmd.MtmdBitmap;
import org.argeo.jjml.mtmd.MtmdContext;
import org.argeo.jjml.mtmd.MtmdImageBitmap;
import org.argeo.jjml.mtmd.MtmdNative;
import org.argeo.jjml.mtmd.MtmdProcessor;

public class ImageDescription {

	public static void main(String[] args) throws Exception {
		MtmdNative.ensureLibrariesLoaded();

		Path modelPath = SimpleModelDownload.getDefaultModelsBase()
				.resolve("ggml-org_Qwen2.5-Omni-3B-GGUF_Qwen2.5-Omni-3B-Q4_K_M.gguf");
		Path mmprojPath = SimpleModelDownload.getDefaultModelsBase()
				.resolve("ggml-org_Qwen2.5-Omni-3B-GGUF_mmproj-Qwen2.5-Omni-3B-Q8_0.gguf");
		Path imagePath = Paths.get(System.getProperty("user.home"), //
				"Pictures/test3.jpg" //
//				"Pictures/test2.png" //
		);
		try (LlamaCppModel model = LlamaCppModel.load(modelPath);
				LlamaCppContext context = new LlamaCppContext(model,
						defaultContextParams().with(ContextParam.n_ctx, 20480)); //
				LlamaCppSamplerChain chain = LlamaCppSamplers.newDefaultSampler(true); //
				MtmdContext mtmdContext = new MtmdContext(model, mmprojPath,
						Runtime.getRuntime().availableProcessors()); //
				InputStream imageIn = Files.newInputStream(imagePath); //
		) {
			MtmdProcessor processor = new MtmdProcessor(context, chain, mtmdContext);
			String prompt = "Describe this image: " + MtmdBackend.getDefaultMarker();
			MtmdImageBitmap bitmap = ImageIoBitmap.load(imageIn);
			MtmdBitmap[] bitmaps = new MtmdBitmap[] { bitmap };
			String response = processor.transcribe(prompt, bitmaps);
			System.out.println(response);
		}
	}

}
