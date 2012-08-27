package dk.blfw.impl;


public interface IntfTrainingProgressProducer {
	void register(IntfTrainingProgressListener listener);
	void unregister(IntfTrainingProgressListener listener);
}
