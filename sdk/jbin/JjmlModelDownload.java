//!/usr/bin/env -S java -cp /usr/share/java/org.argeo.jjml.jar
import org.argeo.jjml.llm.util.SimpleModelDownload;
import org.argeo.jjml.llm.util.SimpleProgressCallback;

/** Downloads a model from Hugging Face. */
public class JjmlModelDownload {

	public static void main(String[] args) throws Exception {
		if (args.length == 0) {
			System.err.println("Download a quantized model (default is Q4_K_M)\n" + //
					"Usage: java " + JjmlModelDownload.class.getSimpleName() + ".java" //
					+ " <hf repo>:[<quantization>]\n" //
					+ "e.g. allenai/OLMo-2-0425-1B-Instruct-GGUF:Q8_0");
		}

		new SimpleModelDownload().getOrDownloadModel(args[0], new SimpleProgressCallback());
	}
}
