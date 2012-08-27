package dk.blfw.stats;

import static dk.blfw.exp.ProjectEnv.Defaults.ARGV;

import java.io.BufferedReader;
import java.io.DataInputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.io.Reader;
import java.text.DecimalFormat;
import java.text.NumberFormat;
import java.util.Formatter;
import java.util.Random;
import java.util.Vector;

import dk.util.math.stats.TestUtils;
import org.apache.commons.math.MathException;
import org.apache.commons.math.stat.inference.TTest;
import org.apache.commons.math.stat.inference.TTestImpl;

import weka.classifiers.Classifier;
import weka.core.Instances;
import weka.core.Summarizable;
import weka.core.Utils;
import dk.blfw.alg.RandomQuerySelector;
import dk.blfw.core.IntfQuerySelector;
import dk.blfw.exp.JobExcecutorBase;
import dk.blfw.exp.ProjectEnv;
import dk.blfw.exp.SimpleJobExecutor;
import dk.blfw.exp.SimpleProgressSummarizer;
import dk.blfw.impl.NaiveBayesAttrUpdatable;
import dk.blfw.impl.QuerySelector;


public class RunAllStats extends JobExcecutorBase {
	
	boolean DEBUG = true;
	String[] dataSets;
	String[] algoNames;
	String[] serFiles;
	String serFileRandom;
	String baselineFile;
	
	int numFolds=10;
	int numDataSets;
	int numAlgos;
	int Budget;
	int numEvalTrials;
	double significanceCutOff = 0.60;

	SimpleProgressSummarizer[][] summarizers;
	SimpleProgressSummarizer[] randomSummarizers;
	double[][][] comparisonMatrix;
	double[][] minBudgetToTarget;
	double[][] dataUtilizationRatio;
	double[][] algoAUC;
	double[] targetMeanErrorRates;
	double[] targetBudgetForRandom;
	double[] baselines;
	int[][] algoRanks;
	int[][][] algoRankRanges;
	double[][] AUCaboveBaselines;

	public SimpleProgressSummarizer getSummarizer(String file) throws Exception{
		return (SimpleProgressSummarizer)SimpleProgressSummarizer.load(null, file);		
	}
	
	public double[][] getTestStatsFromSummarizer(SimpleProgressSummarizer sps){
		return sps.getTrialAccuraciesTest();
	}
	
	public void runAllPairsMedianStat() throws Exception{
		
		for(int d=0; d<numDataSets; d++)
		 for(int i=0; i<numAlgos; i++)
		  for(int j=0; j<numAlgos; j++)
			comparisonMatrix[d][i][j] = runMedianStat(summarizers[d][i], summarizers[d][j]);
    }
	
	public int getBudget(SimpleProgressSummarizer spsA) throws Exception{
		double[][] statsA = getTestStatsFromSummarizer(spsA);
		return statsA.length;
	}

	public double median(double[] Array){
		double[] ArrayCopy = new double[Array.length];
		for(int i=0; i<Array.length; i++)
			ArrayCopy[i] = Array[i];
		//selection sort
		for(int i=0; i<ArrayCopy.length-1; i++) {
		  for (int j=i+1; j<ArrayCopy.length; j++) {
		    if (ArrayCopy[i] > ArrayCopy[j]) {
		     //... Exchange elements
		     double temp = ArrayCopy[i];
		     ArrayCopy[i] = ArrayCopy[j];
		     ArrayCopy[j] = temp;
		    }
		  }
		}		
		if(false)
		{
			for(int i=0; i<ArrayCopy.length; i++)
				System.out.println("SortedArray["+i+"] = "+ArrayCopy[i]);
		}
		int middle = Array.length/2;
		
		if(Array.length % 2 == 0)
			return (ArrayCopy[middle] + ArrayCopy[middle-1]) / 2.0 ;
		else
			return ArrayCopy[middle];
	}
	
	public double meanDifference(double[] A, double[] B){
		
		if(A.length != B.length)
			return Double.NaN;
		double sum = 0.0;
		for(int i=0; i<A.length; i++)
			sum += A[i] - B[i];
		return sum / A.length;
	}
	
	public double runMedianStat(SimpleProgressSummarizer spsA, SimpleProgressSummarizer spsB) throws Exception{
		// test error
		double[][] A = getTestStatsFromSummarizer(spsA);
		double[][] B = getTestStatsFromSummarizer(spsB);
		
		//TTestImpl myTTester = new TTestImpl(); //
		double[] Significances = new double[numEvalTrials];
		for(int i=0; i<numEvalTrials; i++)
		{
			Significances[i] = TestUtils.cPairedTTest(A[i],B[i],0.11111111111111111111);
			if(Double.isNaN(Significances[i]) || Significances[i] < significanceCutOff)
		    	Significances[i] = 0.0;
		    //TODO: Double check that this is correct
			if(meanDifference(A[i], B[i]) > 0.0 && Significances[i] != 0.0) //algo B has the advantage
				Significances[i] = Significances[i] * -1.0;
		}

		if(false)
		{
 		  for(int i=0; i<A.length; i++)
		  {
 			  System.out.println("Trial "+i+":");
 			  System.out.print("A["+i+"] = ");
 			  for(int j=0; j<A[i].length; j++)
 				  System.out.print(A[i][j]+",");
 			  System.out.print("\n");
 			  System.out.print("B["+i+"] = ");
 			  for(int j=0; j<B[i].length; j++)
 				  System.out.print(B[i][j]+",");
 			  System.out.print("\n");
 			  System.out.println("Trial "+i+": Sig = " + Significances[i]); 			  
		  }
		}
		return median(Significances);
	}
	
	public void runRankedPerformance() {

		int[] numBetterThan = new int[numAlgos];

		for(int d=0; d<numDataSets; d++)
		{
			for(int i=0; i<numAlgos; i++)
				numBetterThan[i] = 0;
			for(int i=0; i<numAlgos; i++)
			  for(int j=0; j<numAlgos; j++)
				 if(comparisonMatrix[d][i][j] > 0.0)
					 numBetterThan[i]++;
			for(int i=0; i<numAlgos; i++)
			{
				algoRanks[d][i] = getRank(numBetterThan, i);
				algoRankRanges[d][i][0] = getRankUpperRange(numBetterThan, i);
				algoRankRanges[d][i][1] = getRankLowerRange(numBetterThan, i);
			}

		}
	}
	
	private int getRank(int[] counts, int algo_index)
	{
		int numGreater = 0;
		for(int i=0; i<counts.length; i++)
			if(counts[i] > counts[algo_index])
				numGreater++;
		return numGreater+1;
	}
	
	private int getRankUpperRange(int[] counts, int algo_index)
	{
		int numGreater = 0;
		for(int i=0; i<counts.length; i++)
			if(counts[i] > counts[algo_index])
				numGreater++;
		return numGreater+1;
	}

	private int getRankLowerRange(int[] counts, int algo_index)
	{
		int numGreater = 0;
		int numSame = 0;
		for(int i=0; i<counts.length; i++)
		{
			if(counts[i] > counts[algo_index])
				numGreater++;
			else if(counts[i] == counts[algo_index])
				numSame++;
		}
		return numGreater+numSame+1;
	}
	
	public void runAUCBelowRandom(){
		
		for(int d=0; d<numDataSets; d++)
		{
			double[][] randomCurveArray = getTestStatsFromSummarizer(randomSummarizers[d]);
			for(int i=0; i<numAlgos; i++){
				//Random - Algo mean
				double[][] statsForAlgo = getTestStatsFromSummarizer(summarizers[d][i]);
				double sum = 0.0;
				for(int j=0; j<numEvalTrials; j++)
					sum += Utils.mean(randomCurveArray[j]) - Utils.mean(statsForAlgo[j]);
				algoAUC[d][i] = sum;			
			}
		}		
	}
	
	public void runAUCaboveBaseline() throws IOException{
		
		for(int d=0; d<numDataSets; d++)
		{
			for(int i=0; i<numAlgos; i++){				
				double[][] statsForAlgo = getTestStatsFromSummarizer(summarizers[d][i]);
				double sum = 0.0;
				for(int j=0; j<numEvalTrials; j++)
					sum += Utils.mean(statsForAlgo[j]) - baselines[d];
				AUCaboveBaselines[d][i] = sum;			
			}
		}		
	}

	public void runDataUtilizationRatio(){

		int twentyPercent = (int) (numEvalTrials * .20);
		int windowSize = (int) (numEvalTrials * 0.05);
		boolean found = false;
		int trial_j;

		for(int d=0; d<numDataSets; d++)
		{
			//compute mean error rate of the final 20% of random shopper
			double[][] randomArray = getTestStatsFromSummarizer(randomSummarizers[d]);
			double[] temp_dbl;
			double trialSum = 0.0;
			for(int i=0; i<twentyPercent; i++)  //numEvalTrials-twentyPercent; i<numEvalTrials; i++)
			{
				temp_dbl = randomArray[numEvalTrials-twentyPercent+i];
				trialSum += Utils.mean(temp_dbl);
			}
			targetMeanErrorRates[d] = trialSum / (double) twentyPercent;

			//compute target budget for random shopper 
			double[] tempMeans;
			double[][] statsForRandom = getTestStatsFromSummarizer(randomSummarizers[d]);
			found = false;
			trial_j = windowSize; 
			while(!found)
			{
				//compute budget at which mean of folds reaches targetBudget
				tempMeans = new double[windowSize];
				//for the last (windowSize) trials:
				for(int j=0; j<windowSize; j++)
				{
					//System.out.println("d = "+d+", iteration="+(trial_j-windowSize+j));
				    tempMeans[j] = Utils.mean(statsForRandom[trial_j-windowSize+j]);
				}
				if(Utils.mean(tempMeans) <= targetMeanErrorRates[d])
				{
					targetBudgetForRandom[d] = (double) trial_j;
					found = true;
				}
			    trial_j++;
			    if(trial_j >= numEvalTrials)
			    {
					targetBudgetForRandom[d] = (double) trial_j;
					found = true;
			    }
			}
		    	
			//for original formulation, set windowSize = trial_j in loop below
			for(int i=0; i<numAlgos; i++)
			{
				double[][] statsForAlgo = getTestStatsFromSummarizer(summarizers[d][i]);
				found = false;
				trial_j = windowSize; 
				while(!found)
				{
					tempMeans = new double[windowSize];
					//for the last (windowSize) trials:
					for(int j=0; j<windowSize; j++)
					  tempMeans[j] = Utils.mean(statsForAlgo[trial_j-windowSize+j]);
	
					if(Utils.mean(tempMeans) <= targetMeanErrorRates[d])
					{
						minBudgetToTarget[d][i] = (double) trial_j;
						dataUtilizationRatio[d][i] = minBudgetToTarget[d][i] / (double) targetBudgetForRandom[d];
						found = true;
					}
				    trial_j++;
				    if(trial_j >= numEvalTrials)
				    {
						minBudgetToTarget[d][i] = (double) numEvalTrials;
						dataUtilizationRatio[d][i] = minBudgetToTarget[d][i] / (double) targetBudgetForRandom[d];
						found = true;
				    }
				}
			}
		}
	}
	
	public void runAllStats() throws Exception{
		System.out.println("runAllPairsMedianStat");
		runAllPairsMedianStat();
		System.out.println("runRankedPerformance");
		runRankedPerformance();
		System.out.println("runDataUtilizationRatio");
		runDataUtilizationRatio();
		System.out.println("runAUCBelowRandom");
		runAUCBelowRandom();
		System.out.println("runAUCaboveBaseline");
		runAUCaboveBaseline();
	}
	
	public void printMedianStatisticMatrix(){
		System.out.println("Median Statistic Comparison Matrix:");
		for(int d=0; d<numDataSets; d++)
		{
			System.out.println("DataSet: "+dataSets[d]);
			for(int i=0; i<numAlgos; i++)
			{
				System.out.print("[ ");
				for(int j=0; j<numAlgos; j++)
					System.out.print(comparisonMatrix[d][i][j]+" ");
				System.out.print("]\n");
			}
			System.out.println("\n\n");
		}
	}

	
	public void printRankedPerformanceLatexTable(){

		NumberFormat f = new DecimalFormat ( "#.##" );

		System.out.println("%%%%%%%%%%  Table: Ranked Performances  %%%%%%%%%%");
		System.out.println("\\begin{table*}             %");
		System.out.println("\\caption{Performance Rank ("+numEvalTrials+")} %");
		System.out.println("\\centering                 %");
		System.out.println("\\label{table:PerRank}      %");
		System.out.println("\\begin{tabular}{lllllll}    %");
		System.out.println("\\hline                     %");
		System.out.print("Dataset   ");
		for(int i=0; i<numAlgos; i++)
			System.out.print(" & " + algoNames[i]);
		System.out.print("\\\\ \n");
		System.out.println("\\hline %");
		System.out.println("\\hline %");
		for(int d=0; d<numDataSets; d++)
		{
			System.out.print("\\texttt{"+dataSets[d]+"}  ");
			for(int i=0; i<numAlgos; i++)
			{
				if(algoRankRanges[d][i][0] == algoRankRanges[d][i][1])
					System.out.print(" & "+algoRankRanges[i][0]);
				else
					System.out.print(" & "+algoRankRanges[d][i][0]+"-"+algoRankRanges[d][i][1]);
			}
			System.out.print("  \\\\ \n");
		}
		System.out.println("\\hline                   %");
		System.out.print("Mean ");
		double[] temp_dbl = new double[numDataSets];
		for(int i=0; i<numAlgos; i++)
		{
			System.out.print(" &  ");
			for(int d=0; d<numDataSets; d++)
				temp_dbl[d] = algoRankRanges[d][i][0];
			System.out.format(f.format(Utils.mean(temp_dbl)));
			System.out.print("-");
			for(int d=0; d<numDataSets; d++)
				temp_dbl[d] = algoRankRanges[d][i][1];
			System.out.format(f.format(Utils.mean(temp_dbl)));
		}
		System.out.println("  \\\\");
		System.out.println("Wins BLAH!!!  \\\\");		
		System.out.println("\\hline                   %");
		System.out.println("\\end{tabular}            %");
		System.out.println("\\end{table*}              %");
		System.out.println("\n\n");
	}
	
	public void printTargetBudgetDURLatexTable(){

		NumberFormat f = new DecimalFormat ( "#.##" );
		double[] temp_dbl = new double[numDataSets];

		System.out.println("%%%%%%%%%%  Table: Target Budget  %%%%%%%%%%");
		System.out.println("\\begin{table*}              %");
		System.out.println("\\caption{Target Budget and Data Utilization Rates. ("+numEvalTrials+")}   %");
		System.out.println("\\label{table:TargetBudgetDUR} %");
		System.out.println("\\centering                 %");
		System.out.println("\\begin{tabular}{lllllll}    %");
		System.out.println("\\hline                     %");
		System.out.print("Dataset   ");
		for(int i=0; i<numAlgos; i++)
			System.out.print(" & " + algoNames[i]);
		//System.out.print(" & Target Budget");
		System.out.print("\\\\ \n");
		System.out.println("\\hline %");
		System.out.println("\\hline %");
		for(int d=0; d<numDataSets; d++)
		{
			System.out.print("\\texttt{"+dataSets[d]+"}  ");
			for(int i=0; i<numAlgos; i++)
					System.out.print(" & "+minBudgetToTarget[d][i]);
			System.out.print("  \\\\ \n");
			System.out.print("       ~");
			for(int i=0; i<numAlgos; i++)
			{
				System.out.print(" & (");
				System.out.format(f.format(dataUtilizationRatio[d][i]));
				System.out.print(")");
			}
			System.out.print("  \\\\ \n");
			System.out.println("\\hline %");
		}
//		System.out.print("Median ");
//		for(int i=0; i<numAlgos; i++)
//		{
//			System.out.print(" &  ");
//			for(int d=0; d<numDataSets; d++)
//				temp_dbl[d] = minBudgetToTarget[d][i];
//			System.out.print(median(temp_dbl));
//		}
//		System.out.println("  \\\\");
		System.out.println("\\hline %");
		System.out.print("Mean ");
		for(int i=0; i<numAlgos; i++)
		{
			System.out.print(" &  ");
			for(int d=0; d<numDataSets; d++)
				temp_dbl[d] = minBudgetToTarget[d][i];
			System.out.format(f.format(Utils.mean(temp_dbl)));
		}
		System.out.println("  \\\\");
		System.out.print("Mean DUR ");
		for(int i=0; i<numAlgos; i++)
		{
			System.out.print(" &  ");
			for(int d=0; d<numDataSets; d++)
				temp_dbl[d] = dataUtilizationRatio[d][i];
			System.out.format(f.format(Utils.mean(temp_dbl)));
		}
		System.out.println("  \\\\");
		System.out.print("Median DUR ");
		for(int i=0; i<numAlgos; i++)
		{
			System.out.print(" &  ");
			for(int d=0; d<numDataSets; d++)
				temp_dbl[d] = dataUtilizationRatio[d][i];
			System.out.format(f.format(median(temp_dbl)));
		}
		System.out.println("  \\\\");
		System.out.println("\\end{tabular}            %");
		System.out.println("\\end{table*}              %");
		System.out.println("\n\n");

	}
	
	public void printAACbelowRandomLatexTable(){

		NumberFormat f = new DecimalFormat ( "#.##" );
		double[] temp_dbl = new double[numDataSets];

		System.out.println("%%%%%%%%%%  Table: AAC below Random  %%%%%%%%%%");
		System.out.println("\\begin{table*}               %");
		System.out.println("\\caption{Area Above the Curve Below Random. ("+numEvalTrials+")}   %");
		System.out.println("\\label{table:AACbelowRandom} %");
		System.out.println("\\centering                 %");
		System.out.println("\\begin{tabular}{lllllll}     %");
		System.out.println("\\hline                      %");
		System.out.print("Dataset   ");
		for(int i=0; i<numAlgos; i++)
			System.out.print(" & " + algoNames[i]);
		System.out.print("\\\\ \n");
		System.out.println("\\hline %");
		System.out.println("\\hline %");
		for(int d=0; d<numDataSets; d++)
		{
			System.out.print("\\texttt{"+dataSets[d]+"}  ");
			for(int i=0; i<numAlgos; i++)
			{
				System.out.print(" & ");
				System.out.format(f.format(algoAUC[d][i]));
			}
			System.out.print("  \\\\ \n");
		}
		System.out.println("\\hline                   %");
		System.out.println("\\hline                   %");
		System.out.print("Mean ");
		for(int i=0; i<numAlgos; i++)
		{
			System.out.print(" &  ");
			for(int d=0; d<numDataSets; d++)
				temp_dbl[d] = algoAUC[d][i];
			System.out.format(f.format(Utils.mean(temp_dbl)));
		}
		System.out.print("  \\\\ \n");
		System.out.println("\\end{tabular}            %");
		System.out.println("\\end{table*}              %");
		System.out.println("\n\n");

	}
	
	public void printAUCaboveBaselineLatexTable(){

		NumberFormat f = new DecimalFormat ( "#.##" );
		double[] temp_dbl = new double[numDataSets];

		System.out.println("%%%%%%%%%%  Table: AUC-baseline  %%%%%%%%%%");
		System.out.println("\\begin{table*}               %");
		System.out.println("\\caption{Area Under the Learning Curve Above Baseline. ("+numEvalTrials+")}   %");
		System.out.println("\\label{table:AUCaboveBaseline} %");
		System.out.println("\\centering                 %");
		System.out.println("\\begin{tabular}{lllllll}     %");
		System.out.println("\\hline                      %");
		System.out.print("Dataset   ");
		for(int i=0; i<numAlgos; i++)
			System.out.print(" & " + algoNames[i]);
		System.out.print("\\\\ \n");
		System.out.println("\\hline %");
		System.out.println("\\hline %");
		for(int d=0; d<numDataSets; d++)
		{
			System.out.print("\\texttt{"+dataSets[d]+"}  ");
			for(int i=0; i<numAlgos; i++)
			{
				System.out.print(" & ");
				System.out.format(f.format(AUCaboveBaselines[d][i]));
			}
			System.out.print("  \\\\ \n");
		}
		System.out.println("\\hline                   %");
		System.out.println("\\hline                   %");
		System.out.print("Mean ");
		for(int i=0; i<numAlgos; i++)
		{
			System.out.print(" &  ");
			for(int d=0; d<numDataSets; d++)
				temp_dbl[d] = AUCaboveBaselines[d][i];
			System.out.format(f.format(Utils.mean(temp_dbl)));
		}
		System.out.print("  \\\\ \n");
		System.out.println("\\end{tabular}            %");
		System.out.println("\\end{table*}              %");
		System.out.println("\n\n");

	}
	
	public void printSummary(){
		
		System.out.println("DEBUG: targetMeanErrorRates, targetBudgetForRandom:");
		for(int i=0; i<numDataSets; i++)
			System.out.println("ds["+i+"]: "+targetMeanErrorRates[i]+", "+targetBudgetForRandom[i]);
		
		
		System.out.println("Statistics Summary");
		System.out.println("=============================");
		System.out.println(" Budget:           "+Budget);
		System.out.println(" Trials Evaluated: "+numEvalTrials);
		System.out.println(" Datasets:  (baselines)");
		for(int i=0; i<numDataSets; i++)
			System.out.println(" "+dataSets[i]+" ("+baselines[i]+")");
		System.out.println("Algorithms: ");
		for(int i=0; i<numAlgos; i++)
			System.out.println(" "+algoNames[i]);
		System.out.println("\n\n");
		
		//printMedianStatisticMatrix();
		//printRankedPerformanceLatexTable();
		printTargetBudgetDURLatexTable();
		printAACbelowRandomLatexTable();
		printAUCaboveBaselineLatexTable();
	}

	public static void main(String[] argv)throws Exception
	{
		RunAllStats myStats = new RunAllStats();
		ProjectEnv env= new ProjectEnv();
		env.put(ARGV, argv);
		myStats.setEnv(env);
		System.out.println("initilaize");
		myStats.initialize();
		System.out.println("runAllStats");
		myStats.runAllStats();
		System.out.println("printSummary");
		myStats.printSummary();
				
	}

	public void initialize() throws IOException  {
		String[] argv= (String []) env.get(ARGV);
		
		
		try {
			if(argv.length==0||Utils.getFlag('h',argv)) {
				throw new Exception("Usage: RunStats -T budget "+
						                            "-datasets  [\"DataSet Names\"]" + 
						                            "-algoNames [\"list of algorithm Names\"]" +
						                            "-serFileRandomPrefix [ser File prefix for RandomQuerySelector]" +
						                            "-baselineFilePrefix [prefix for baseline flat file]" +
						                            "-serFilePrefixes  [\"list of ser Files\"]\n");
				
			}

			String listOfDataSets=Utils.getOption("datasets", argv);
			if (listOfDataSets.length()!=0)
				 dataSets = listOfDataSets.split("\\s+");
			
			String list_of_algos=Utils.getOption("algoNames", argv);
			if (list_of_algos.length()!=0)
				 algoNames = list_of_algos.split("\\s+");
									
			String list_of_serFiles=Utils.getOption("serFilePrefixes", argv);
			if (list_of_serFiles.length()!=0)
				 serFiles = list_of_serFiles.split("\\s+");

			serFileRandom = Utils.getOption("serFileRandomPrefix", argv);
			baselineFile  = Utils.getOption("baselineFilePrefix", argv);

			String cmd_line_budget = Utils.getOption("T", argv);
			if(cmd_line_budget.length()!=0)
				numEvalTrials = Integer.parseInt(cmd_line_budget);
			else
				numEvalTrials = -1;
				
		} catch (Exception e) {
			
			e.printStackTrace();
		}

		numDataSets = dataSets.length;
		numAlgos = algoNames.length;

		summarizers = new SimpleProgressSummarizer[numDataSets][numAlgos];
		randomSummarizers = new SimpleProgressSummarizer[numDataSets];
		String serFileName = null;
		for(int i=0; i<numDataSets; i++)
		  for(int j=0; j<numAlgos; j++)
		  {
			try {
				serFileName = serFiles[j]+"_"+dataSets[i]+".ser";
				System.out.println("Opening: "+serFileName);
				summarizers[i][j] = getSummarizer(serFileName);
			} catch (Exception e) {
				// TODO Auto-generated catch block
				System.err.println("Error in loading ser file: "+serFileName);
				e.printStackTrace();
			}
		  }

			
		for(int i=0; i<numDataSets; i++)
		{
		  try { 
				serFileName = serFileRandom+dataSets[i]+".ser";
				System.out.println("Opening: "+serFileName);
			    randomSummarizers[i] = getSummarizer(serFileName);
		} catch (Exception e) {
			// TODO Auto-generated catch block
			System.err.println("Error in loading ser file: "+serFileName);
			e.printStackTrace();
		}
		}

		try {
			Budget = getBudget(randomSummarizers[0]);
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		if(numEvalTrials == -1)
			numEvalTrials = Budget;

		comparisonMatrix = new double[numDataSets][numAlgos][numAlgos];
		minBudgetToTarget = new double[numDataSets][numAlgos];
		dataUtilizationRatio = new double[numDataSets][numAlgos];
		algoRanks = new int[numDataSets][numAlgos];
		algoRankRanges = new int[numDataSets][numAlgos][2];
		algoAUC = new double[numDataSets][numAlgos];
		AUCaboveBaselines = new double[numDataSets][numAlgos];

		baselines = new double[numDataSets];
		for(int d=0; d<numDataSets; d++)
		{
			String baselineFileName = baselineFile+"_"+dataSets[d]+".graph.output";
			BufferedReader br = new BufferedReader(new FileReader(baselineFileName));
			String line = br.readLine();
			String[] tempDbls = line.split(",");
			baselines[d] = Double.valueOf(tempDbls[3]);
		}

		for(int k=0; k<numDataSets; k++)
		 for(int i=0; i<numAlgos; i++)
		  for(int j=0; j<numAlgos; j++)
			comparisonMatrix[k][i][j] = -1.0;
		targetMeanErrorRates = new double[numDataSets];
		targetBudgetForRandom = new double[numDataSets];
		for(int d=0; d<numDataSets; d++)
		 for(int i=0; i<numAlgos; i++)
		 {
			algoRanks[d][i] = -1;
			algoRankRanges[d][i][0] = -1;
			algoRankRanges[d][i][1] = -1;
			minBudgetToTarget[d][i] = -1.0;
			dataUtilizationRatio[d][i] = -1.0;
			algoAUC[d][i] = -1.0;
		 }
	}

	public void run() {
		// TODO Auto-generated method stub
		
	}
}
