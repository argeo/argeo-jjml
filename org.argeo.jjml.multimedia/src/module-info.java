module org.argeo.jjml.multimedia {
	exports org.argeo.jjml.imageio.mtmd;
	exports org.argeo.jjml.sound.whisper;

	requires transitive java.desktop;

	requires transitive org.argeo.jjml;
}