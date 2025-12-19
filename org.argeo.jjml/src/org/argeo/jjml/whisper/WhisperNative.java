package org.argeo.jjml.whisper;

import org.argeo.jjml.llm.LlamaCppNative;

/** Availability of the native bindings to whisper.cpp. */
public class WhisperNative {
	private final static String JJML_WHISPER_LIBRARY_NAME = "Java_org_argeo_jjml_whisper";

	private static boolean librariesLoaded = false;

	/*
	 * STATIC UTILITIES
	 */
	public static boolean isAvailable() {
		try {
			ensureLibrariesLoaded();
			return true;
		} catch (UnsatisfiedLinkError e) {
			return false;
		}
	}

	public synchronized static void ensureLibrariesLoaded() {
		if (librariesLoaded)
			return;
		LlamaCppNative.ensureLibrariesLoaded();
		loadLibraries();
	}

	synchronized static void loadLibraries() {
		checkLibrariesNotLoaded();
		System.loadLibrary(JJML_WHISPER_LIBRARY_NAME);
		librariesLoaded = true;
	}

	/** Fails if libraries already loaded. */
	private static void checkLibrariesNotLoaded() {
		if (librariesLoaded)
			throw new IllegalStateException("Shared libraries are already loaded.");
	}

	/** singleton */
	private WhisperNative() {
	}
}
