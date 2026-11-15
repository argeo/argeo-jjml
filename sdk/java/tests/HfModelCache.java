package tests;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;

/**
 * Downloads a GGUF model (by defaults from HuggingFace) with the same naming
 * conventions as llama-cli. This is meant to be used for prototyping, not as a
 * full-fledged models management solution.
 */
public class HfModelCache {
	private final Path modelsBase;

	HfModelCache(Path modelsBase) {
		this.modelsBase = modelsBase;
	}

	public HfModelCache() {
		this(getDefaultModelsBase());
	}

	private Path getLocalHfRepoBaseDir(String hfRepo) {
		return modelsBase.resolve("models--" + hfRepo.replace("/", "--"));
	}

	/*
	 * LLM
	 */
	public Path getLocalFile(String hfRepoArg) throws IOException {
		if (hfRepoArg.contains(":")) {
			return getLocalFile(hfRepoArg.split(":")[0], hfRepoArg.split(":")[1]);
		} else {
			return getLocalFile(hfRepoArg, "Q4_K_M");
		}
	}

	public Path getLocalFile(String hfRepo, String quantization) throws IOException {
		Path baseDir = getLocalHfRepoBaseDir(hfRepo);
		Path refsFile = baseDir.resolve("refs").resolve("main");
		if (!Files.exists(refsFile))
			return null;
		String ref = Files.readString(refsFile).strip();
		Path modelsDir = baseDir.resolve("snapshots").resolve(ref);
		if (!Files.exists(modelsDir))
			throw new IllegalArgumentException(
					modelsDir + " does not exist, make sure the model has been downloaded already");
		Path path = modelsDir.resolve(getLocalFileName(hfRepo, quantization, "-"));
		if (!Files.exists(path))
			path = modelsDir.resolve(getLocalFileName(hfRepo, quantization, "."));
		if (!Files.exists(path))
			throw new IllegalArgumentException("Cannot find quantization " + quantization + " in " + modelsDir);
		return path.toRealPath();
	}

	private String getLocalFileName(String hfRepo, String quantization, String quantSep) {
		String fileName = hfRepo.split("/")[1].replace("-GGUF", quantSep + quantization + ".gguf");
		String localFileName = fileName;
		return localFileName;
	}

	/*
	 * MMPROJ
	 */
	public Path getLocalMmprojFile(String hfRepoArg) throws IOException {
		if (hfRepoArg.contains(":")) {
			return getLocalMmprojFile(hfRepoArg.split(":")[0], hfRepoArg.split(":")[1]);
		} else {
			return getLocalMmprojFile(hfRepoArg, "BF16");
		}
	}

	public Path getLocalMmprojFile(String hfRepo, String quantization) throws IOException {
		// TODO factorize
		Path baseDir = getLocalHfRepoBaseDir(hfRepo);
		Path refsFile = baseDir.resolve("refs").resolve("main");
		if (!Files.exists(refsFile))
			return null;
		String ref = Files.readString(refsFile).strip();
		Path modelsDir = baseDir.resolve("snapshots").resolve(ref);
		return modelsDir.resolve(getLocalMmprojFileName(hfRepo, quantization)).toRealPath();
	}

	private String getLocalMmprojFileName(String hfRepo, String quantization) {
		String fileName = hfRepo.split("/")[1].replace("-GGUF", "-" + quantization + "-mmproj.gguf");
		String localFileName = fileName;
		return localFileName;
	}

	/*
	 * STATIC
	 */
	/** The default path where GGUF files are downloaded and searched for. */
	private static Path getDefaultModelsBase() {
		Path defaultModelsBase;
		String os = System.getProperty("os.name").toLowerCase();
		if (os.contains("win")) {
			defaultModelsBase = Paths.get(System.getProperty("user.home"), ".cache", "huggingface", "hub");
		} else if (os.contains("mac") || os.contains("darwin")) {
			defaultModelsBase = Paths.get(System.getProperty("user.home"), ".cache", "huggingface", "hub");
		} else { // Linux / Unix
			defaultModelsBase = Paths.get(System.getProperty("user.home"), ".cache", "huggingface", "hub");
		}
		return defaultModelsBase;
	}

}
