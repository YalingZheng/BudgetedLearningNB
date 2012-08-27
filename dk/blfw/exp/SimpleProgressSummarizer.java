package dk.blfw.exp;

import java.io.BufferedInputStream;
import java.io.BufferedOutputStream;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.OutputStream;
import java.io.PrintStream;
import java.io.Serializable;
import java.util.Vector;
import java.util.zip.GZIPInputStream;
import java.util.zip.GZIPOutputStream;

import weka.classifiers.Classifier;
import weka.classifiers.evaluation.ConfusionMatrix;
import weka.classifiers.evaluation.EvaluationUtils;
import weka.core.Attribute;
import weka.core.FastVector;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Utils;

import dk.blfw.impl.IntfTrainingProgressListener;

public class SimpleProgressSummarizer implements IntfTrainingProgressListener, Serializable{
	
	/**
	 * Cleaning up...
	 */
	private static final long serialVersionUID = 1L;
	private String name="";	
	private String row_selector="";
	private double gamma=0.0;
	Instances summary= null;	
	private Instances results = null;	
	int fn=0;
	int qn=0;
	double train_err=0, err_train=0, err_test=0;
	Instances trainFold, testFold;
	String[] classnames= null;
	private int numFolds = 10;
	private int numTrials = 500;
	double[][] trialAccuraciesTest;
	double[][] trialAccuraciesTrain;
	
	public static void save(Object o, String outFile) throws Exception{
	    FileOutputStream ostream = new FileOutputStream(outFile);
	    OutputStream os = ostream;
	    ObjectOutputStream p;
	    
	    p = new ObjectOutputStream(new BufferedOutputStream(new GZIPOutputStream(os)));
	    p.writeObject(o);
	    p.flush();
	    p.close(); // used to be ostream.close() !
	}
	
	public static Object load(Object o, String inFile) throws Exception{
	    try {
	        FileInputStream istream = new FileInputStream(inFile);
	        ObjectInputStream p;
	  	    p = new ObjectInputStream(new BufferedInputStream(new GZIPInputStream(istream)));
	        Object toReturn = p.readObject();
	        istream.close();
	        return toReturn;
	      } catch (Exception e) {
	    	  e.printStackTrace();
	        return null;
	      }
	}

	public void init(){
		qn=0;
		fn++;
	}
	
	public void done() {
		//trainFold=null;
		//testFold=null;
	}

	public SimpleProgressSummarizer(String name, int numTrials, int numFolds){
		this.numTrials = numTrials;
		this.numFolds = numFolds;
		//System.out.println("numTrials="+numTrials);
		//System.out.println("numFolds="+numFolds);
		trialAccuraciesTrain = new double[numTrials][numFolds];
		trialAccuraciesTest = new double[numTrials][numFolds];
		this.name=name;
		// generate pseudo-dataset with instance ids, to get the same reordering..
		summary = createInstances();
		results = createInstances();
	}
	
	public SimpleProgressSummarizer(){
		this("anonymous", 500, 10);
	}
	
	private Instances createInstances() {
		FastVector attInfo = new FastVector(5);

		attInfo.addElement(new Attribute("FoldNum"));
		attInfo.addElement(new Attribute("QueryNum"));
		attInfo.addElement(new Attribute("training_acc"));
		attInfo.addElement(new Attribute("acc_train"));
		attInfo.addElement(new Attribute("acc_test"));

		Instances obj = new Instances("summary",attInfo,10000);
		obj.setClassIndex(-1);
		return obj;
	}
	
	public void update(Classifier c, Instances train, int num) {
		
		if (classnames==null){
			classnames= new String[train.classAttribute().numValues()];
			for(int i=0; i<classnames.length; i++){
				classnames[i]= "class"+ Integer.toString(i);
			}
		}
		
		EvaluationUtils eval= new EvaluationUtils();
		try {
			FastVector vecTrain=eval.getTestPredictions(c, train);
			FastVector vecFoldTrain=eval.getTestPredictions(c, trainFold);
			FastVector vecFoldTest= eval.getTestPredictions(c, testFold);
			
			ConfusionMatrix cmTrain= new ConfusionMatrix(classnames);
			ConfusionMatrix cmFoldTrain= new ConfusionMatrix(classnames);
			ConfusionMatrix cmFoldTest = new ConfusionMatrix(classnames);
			
			cmTrain.addPredictions(vecTrain);
			cmFoldTrain.addPredictions(vecFoldTrain);
			cmFoldTest.addPredictions(vecFoldTest);
			
			double e= cmTrain.errorRate();
			double etrain= cmFoldTrain.errorRate();
			double etest= cmFoldTest.errorRate();
			//fold number, trial number, error on partial train, error on train, error on test
			double[] o= new double[]{fn, num, e, etrain, etest};
			Instance inst= new Instance(1,o);
			
			inst.setDataset(summary);
			summary.add(inst);
			
			trialAccuraciesTest[num-1][fn-1] = etest;
			trialAccuraciesTrain[num-1][fn-1] = etrain;
			results.add(inst);
			
		} catch (Exception e) {
			e.printStackTrace();
		}
		
		
	}
	
	public void setFold(Instances train, Instances test){
		trainFold=train;
		testFold=test;
		
	}
	
	public void print(PrintStream stream){
		stream.print(summary.toString());
	}
	
	public void printSummary(PrintStream stream){
		
		double[][] err;
		double[][] err_train;
		double[][] err_test;
		err = new double[numTrials][numFolds];
		err_train = new double[numTrials][numFolds];
		err_test = new double[numTrials][numFolds];
		Instance inst;
		for(int i=0; i<results.numInstances(); i++)
		{
			inst = results.instance(i);
			err[(int)inst.value(1)-1][(int)inst.value(0)-1] = inst.value(2);
			err_train[(int)inst.value(1)-1][(int)inst.value(0)-1] = inst.value(3);
			err_test[(int)inst.value(1)-1][(int)inst.value(0)-1] = inst.value(4);
		}

		//System.out.println("results has "+results.numInstances()+" instances");
		for(int i=0; i<numTrials; i++)
		{
		  double mean_err, mean_err_train, mean_err_test;
		  mean_err= Utils.mean(err[i]);
		  mean_err_train= Utils.mean(err_train[i]);
		  mean_err_test= Utils.mean(err_test[i]);
		  stream.println(i + "," + mean_err + "," + mean_err_train + ","+ mean_err_test);		
		}

//		for(int i=0; i<err_test.length; i++)
//		{
//			System.out.print("Trial "+i+": ");
//			for(int j=0; j<err_test[i].length; j++)
//			{
//				System.out.print(err_test[i][j] + ", ");				
//			}
//			System.out.print("\n");
//		}		

		//		for(int i=0; i<vec_query_fold.size();i++){
//			Instances vec_fold = vec_query_fold.get(i);
//			double[] err= new double[vec_fold.numInstances()];
//			double[] err_train= new double[vec_fold.numInstances()];
//			double[] err_test= new double[vec_fold.numInstances()];
//			
//			for (int j=0; j< err.length; j++){
//				err[j]=vec_fold.instance(j).value(2);
//				err_train[j]=vec_fold.instance(j).value(3);
//				err_test[j]=vec_fold.instance(j).value(4);
//			}
//			double mean_err, mean_err_train, mean_err_test;
//			mean_err= Utils.mean(err);
//			mean_err_train= Utils.mean(err_train);
//			mean_err_test= Utils.mean(err_test);
			
	}
	
	public double[][] getTrialAccuraciesTest(){
		return trialAccuraciesTest;
	}
	public double[][] getTrialAccuraciesTrain(){
		return trialAccuraciesTrain;
	}

	public String getName() {
		return name;
	}

	public void setNumFolds(int numFolds) {
		this.numFolds = numFolds;
	}

	public int getNumFolds() {
		return numFolds;
	}

	public void setNumTrials(int numTrials) {
		this.numTrials = numTrials;
	}

	public int getNumTrials() {
		return numTrials;
	}

	public void setRowSelector(String row_selector) {
		this.row_selector = row_selector;
	}

	public String getRowSelector() {
		return row_selector;
	}

	public void setGamma(double gamma) {
		this.gamma = gamma;
	}

	public double getGamma() {
		return gamma;
	}
}
	



