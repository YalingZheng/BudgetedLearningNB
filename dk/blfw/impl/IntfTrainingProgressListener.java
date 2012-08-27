package dk.blfw.impl;

import weka.classifiers.Classifier;
import weka.core.Instances;

public interface IntfTrainingProgressListener {
	void init();
	void update(Classifier c, Instances train, int num);
	void done();
}
