package tests.whisper;

import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Objects;

import tests.AbstractJjmlTests;

abstract class AbstractWhisperTests extends AbstractJjmlTests {
	private String modelHint;

	public AbstractWhisperTests(String modelHint) {
		Objects.requireNonNull(modelHint);
		this.modelHint = modelHint;
	}

	protected Path getModelPath() {
		Path modelPath = Paths.get(System.getProperty("user.home"), "dev/foss/AI/openai/converted", modelHint);
		return modelPath;
	}
}
