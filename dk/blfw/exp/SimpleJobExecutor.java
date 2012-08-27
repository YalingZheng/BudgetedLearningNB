package dk.blfw.exp;

import java.io.FileReader;
import java.util.Random;


import dk.blfw.alg.BudgetedLearner;
import dk.blfw.alg.RandomQuerySelector;
import dk.blfw.core.IntfBudget;
import dk.blfw.core.IntfOracle;
import dk.blfw.core.IntfQuerySelector;
import dk.blfw.impl.NaiveBayesAttrUpdatable;
import dk.blfw.impl.PhonyAttrOracle;
import dk.blfw.impl.PhonySimpleBudget;
import dk.blfw.impl.QuerySelector;
import weka.classifiers.Classifier;
import weka.core.*;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.ReplaceMissingValues;
import static dk.blfw.exp.ProjectEnv.Defaults.*;

public class SimpleJobExecutor extends JobExcecutorBase {
	
	public void run() {
		String[] argv= (String []) env.get(ARGV);

		try {
			int numFolds=0; int cIdx=-1; int seed=1; int budget=500;

			if (argv.length==0||Utils.getFlag('h',argv)) {
				throw new Exception(
						"Usage: SimpleJobExecutor  -base [BaseClassifier] -basearg [\"argsForbase \"]" + 
						"-q [QuerySelector] -qarg [\" argsForq \"]  -t [TrainFile] [-T [TestFile]]  " +
						"[-x numFolds] [-s randomSeed] [-c classIndex] [-b budget] [-e delimit] \n  " +
						"[-o outFile]" +
						"outputs testing error for each budget, where total budget is specified by -b option. \n       " +
						"If no test file is given, does a cross-validation on training data.\n       " +
						"Format: FoldID Budget TrainingError TestingError \n       " +
						"If outFile is given, the intermediate results for each fold is written to that file"+
				"Field delimiter can be changed via -e (default: space)\n");
			}


			String classifierName=Utils.getOption("base", argv);
			if (classifierName.length()==0) {
				classifierName= NaiveBayesAttrUpdatable.class.getName();
			}
			
			Classifier base;
			String argsForBase=Utils.getOption("basearg", argv);
			if (argsForBase.length()!=0) {
				 String[] args=  argsForBase.split("\\s");
				 base = Classifier.forName(classifierName,args);
				
			}else {
				 base=Classifier.forName(classifierName,null);
			}
			

			String querySelectorName= Utils.getOption('q', argv);
			if (querySelectorName.length()==0){
				querySelectorName= RandomQuerySelector.class.getName();
			}
			
			
			IntfQuerySelector qs;
			String argsForQS=Utils.getOption("qarg", argv);
			if (argsForQS.length()!=0) {
				 String[] args=  argsForQS.split("\\s");
				 qs= QuerySelector.forName(querySelectorName,args);
				
			}else {
				 qs=  QuerySelector.forName(querySelectorName,null);
			}
			

			String delim = Utils.getOption('e',argv);
			if (delim.length()==0) {
				delim=" ";
			}


			String cv = Utils.getOption('x',argv);
			if (cv.length()!=0) {
				numFolds=Integer.parseInt(cv);
			} else {
				numFolds=10; // default
			}

			String seedS = Utils.getOption('s',argv);
			if (seedS.length()!=0) {
				seed=Integer.parseInt(seedS);
			}

			String trainFile = Utils.getOption('t',argv);
						
			Instances trainData = new Instances(new FileReader(trainFile));
			if (trainFile.length()==0) {
				   throw new Exception("No train file given!");
			}
			

			String testFile  = Utils.getOption('T',argv);			
			
			Instances testData = null;

			String classIdx = Utils.getOption('c',argv);
			if (classIdx.length()!=0) {
				cIdx=Integer.parseInt(classIdx)-1;
				if ((cIdx<0)||(cIdx>=trainData.numAttributes())) throw new Exception("Invalid value for class index!");
			} else {
				cIdx=trainData.numAttributes()-1;
			}

			if (testFile.length()!=0)
				testData  = new Instances(new FileReader(testFile));	        

			
			trainData.setClassIndex(cIdx);
			

			String budgetS= Utils.getOption('b', argv);
			if (budgetS.length()!=0){
				budget=Integer.parseInt(budgetS);
			}

			String outFile= Utils.getOption('o', argv);
			


			if (testData==null) {
				if (numFolds<2||numFolds>trainData.numInstances()) {
					throw new Exception("Invalid number of cross-validation folds!");
				}

				//int numFolds=0; int cIdx=-1; int seed=1; int budget=0; String delim= " "; 
				runOnTrainData(base, qs, trainData, cIdx, numFolds, seed, budget,delim,outFile);
			}else{
				throw new Exception("not implemented yet!");
			}

		} catch (Exception e) {
			e.printStackTrace(System.err);
		}
	}


	private void runOnTrainData(Classifier base, IntfQuerySelector qs, Instances trainData, int cIdx, int numFolds, int seed, double budget, String delim, String outFile) throws Exception{
		
		//fill in missing attributes with mean and mode
	
	
			Filter filter = new ReplaceMissingValues();
			filter.setInputFormat(trainData);
			for (int i = 0; i < trainData.numInstances(); i++) {
				filter.input(trainData.instance(i));
			}
			filter.batchFinished();
		
			trainData = filter.getOutputFormat();
			Instance processed;
			while ((processed = filter.output()) != null) {
				trainData.add(processed);
			}
		

			// generate pseudo-dataset with instance ids, to get the same reordering..
			FastVector attInfo = new FastVector(2);

			attInfo.addElement(new Attribute("Idx_20011004"));
			attInfo.addElement(trainData.classAttribute());

			Instances indices = new Instances("Indices",attInfo,trainData.numInstances());
			indices.setClass((Attribute)attInfo.elementAt(1));

			for (int k = 0; k < trainData.numInstances(); k++) {
				Instance inst = new Instance(2);
				inst.setDataset(indices);
				inst.setClassValue(trainData.instance(k).classValue());
				inst.setValue(0,k);
				indices.add(inst);

				Random random = new Random(seed);
				random.setSeed(seed);
				indices.randomize(random);

				random = new Random(seed);
				random.setSeed(seed);
				trainData.randomize(random);

				if (trainData.classAttribute().isNominal()) {
					trainData.stratify(numFolds);
					indices.stratify(numFolds);
				}
			}
		
			SimpleProgressSummarizer summarizer= new SimpleProgressSummarizer(qs.getName(),(int)budget,numFolds);

			//deng: how to do these?
			//summarizer.setRowSelector("Random");
			summarizer.setGamma(0.0);
		
			
			dk.blfw.Global.random.setSeed(seed);
			System.out.println("BEGIN------------");
			for (int i=0; i<numFolds; i++) {
			
				//System.out.println("seed="+dk.blfw.Global.permutationseed);
				BudgetedLearner bl= new BudgetedLearner();
				IntfBudget ub= new PhonySimpleBudget(budget);
				Instances train = trainData.trainCV(numFolds,i);
				Instances test = trainData.testCV(numFolds,i);
				summarizer.setFold(train, test);
				IntfOracle o= new PhonyAttrOracle(train);
			
				//make a copy for base
				Classifier baseCopy=Classifier.makeCopy(base);
			
				//make a copy for qs
				IntfQuerySelector qsCopy= QuerySelector.makeCopy(qs);
			
				bl.initialize(baseCopy,ub,o,qsCopy);
				bl.register(summarizer);			//todo:...
				bl.buildClassifier(train);
			}
		
			summarizer.printSummary(System.out);
		
			if (outFile.length()!=0) {
				SimpleProgressSummarizer.save(summarizer, outFile);
			}
	}
	

	

	/**
	 * @param args
	 */
	public static void main(String[] argv) {
		SimpleJobExecutor sje= new SimpleJobExecutor();
		ProjectEnv env= new ProjectEnv();
		env.put(ARGV, argv);
		sje.setEnv(env);
		sje.run();
	}
}
