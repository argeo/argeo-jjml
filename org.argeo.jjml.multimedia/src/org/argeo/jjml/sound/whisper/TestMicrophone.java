package org.argeo.jjml.sound.whisper;

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.IOException;

import javax.sound.sampled.AudioFileFormat;
import javax.sound.sampled.AudioFormat;
import javax.sound.sampled.AudioInputStream;
import javax.sound.sampled.AudioSystem;
import javax.sound.sampled.Clip;
import javax.sound.sampled.DataLine;
import javax.sound.sampled.Line;
import javax.sound.sampled.LineUnavailableException;
import javax.sound.sampled.Mixer;
import javax.sound.sampled.TargetDataLine;

public class TestMicrophone {
	static final int totalSize = 16000 * 10;
	static int totalWritten = 0;
	static AudioFormat format = new AudioFormat(8000.0f, 16, 1, true, true);

	void doIt() throws Exception {
		TargetDataLine line = getTargetDataLineForRecord();
		int frameSizeInBytes = format.getFrameSize();
		int bufferLengthInFrames = line.getBufferSize() / 8;
		final int bufferLengthInBytes = bufferLengthInFrames * frameSizeInBytes;

//		buildByteOutputStream(out, line, frameSizeInBytes, bufferLengthInBytes);
//		this.audioInputStream = new AudioInputStream(line);
//
//		setAudioInputStream(convertToAudioIStream(out, frameSizeInBytes));
//		audioInputStream.reset();
	}

	public static void main(String[] args) throws Exception {
//		TestMicrophone testMicrophone = new TestMicrophone();
//		testMicrophone.doIt();

		TargetDataLine microphone = null;
//		Mixer.Info[] mixerInfos = AudioSystem.getMixerInfo();
//		for (Mixer.Info info : mixerInfos) {
//			Mixer m = AudioSystem.getMixer(info);
//			Line.Info[] lineInfos = m.getSourceLineInfo();
//			for (Line.Info lineInfo : lineInfos) {
////				System.out.println(info.getName() + "---" + lineInfo);
////				Line line = m.getLine(lineInfo);
////				System.out.println("\t-----" + line);
//			}
//			lineInfos = m.getTargetLineInfo();
//			for (Line.Info lineInfo : lineInfos) {
////				System.out.println(m + "---" + lineInfo);
////				Line line = m.getLine(lineInfo);
////				System.out.println("\t-----" + line);
//				System.out.println(info.getName() + "---" + lineInfo);
//				String targetName="ALSA Playback [default]";
////				 targetName="Port Headset [hw:2]";
//				if(targetName.equals(info.getName())) {
//					microphone =(TargetDataLine) m.getLine(lineInfo);
//				}
//			}
//
//		}
//
		ByteArrayOutputStream out = new ByteArrayOutputStream(totalSize);

		AudioFormat format = new AudioFormat(16000.0f, 16, 1, true, true);
		microphone = AudioSystem.getTargetDataLine(format);

		int frameSizeInBytes = format.getFrameSize();
		int bufferLengthInFrames = microphone.getBufferSize() / 8;
		final int bufferLengthInBytes = bufferLengthInFrames * frameSizeInBytes;

		microphone.open(format, microphone.getBufferSize());
		System.out.println(microphone.getLineInfo());

		microphone.start();
		Thread.sleep(1000);
//		byte[] buf = new byte[microphone.available()];
//		microphone.read(buf, 0, buf.length);
		buildByteOutputStream(out, microphone, bufferLengthInBytes);
		microphone.stop();
		System.out.println("Stopped recording");

		byte[] buf = out.toByteArray();
		AudioInputStream ais = new AudioInputStream(new ByteArrayInputStream(buf), format,
				buf.length / format.getFrameSize());
		File outFile = new File("recorded.wav");
		AudioSystem.write(ais, AudioFileFormat.Type.WAVE, outFile);
		ais.close();
		System.out.println("Wrote " + outFile);

//		Clip clip = AudioSystem.getClip();
//		clip.open(ais);
//
//		Thread.sleep(4000);
	}

	private TargetDataLine getTargetDataLineForRecord() throws LineUnavailableException {
		TargetDataLine line;
		DataLine.Info info = new DataLine.Info(TargetDataLine.class, format);
		if (!AudioSystem.isLineSupported(info)) {
			return null;
		}
		line = (TargetDataLine) AudioSystem.getLine(info);
		line.open(format, line.getBufferSize());
		return line;
	}

	static void buildByteOutputStream(final ByteArrayOutputStream out, final TargetDataLine line,
			final int bufferLengthInBytes) throws IOException {
		final byte[] data = new byte[bufferLengthInBytes];
		int numBytesRead;
		// line.start();
		while (true) {
			if ((numBytesRead = line.read(data, 0, bufferLengthInBytes)) == -1) {
				break;
			}
			out.write(data, 0, numBytesRead);
			totalWritten = totalWritten + numBytesRead;
			System.out.println("Read " + totalWritten + " bytes");
			if (totalWritten > totalSize)
				break;
		}
	}

	public AudioInputStream convertToAudioIStream(final ByteArrayOutputStream out, int frameSizeInBytes) {
		byte audioBytes[] = out.toByteArray();
		ByteArrayInputStream bais = new ByteArrayInputStream(audioBytes);
		AudioInputStream audioStream = new AudioInputStream(bais, format, audioBytes.length / frameSizeInBytes);
//	    long milliseconds = (long) ((audioStream.getFrameLength() * 1000) / format.getFrameRate());
//	    duration = milliseconds / 1000.0;
		return audioStream;
	}
}
