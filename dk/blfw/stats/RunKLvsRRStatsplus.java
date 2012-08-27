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


public class RunKLvsRRStatsplus extends JobExcecutorBase {
	
	String serFileDirectory = "./10itersoutput/";
	boolean DEBUG = true;
	String[] dataSets;
	String[] algoNames;
	String[] serFilePrefixesRR;
	String[] serFilePrefixesKL;
	
	// Number of Iterations
	int numIter = 10;
	int numFolds = 10;
	int numTrials = 500;
	int numDataSets;
	int numAlgos;
	int Budget;
	int numEvalTrials;
	String category;

	SimpleProgressSummarizer[][][] randomRowSummarizers;
	SimpleProgressSummarizer[][][] KLDivergenceSummarizers;
	String[][][] serFiles_RR;
	String[][][] serFiles_KL;
	
	double[][] KLvsRRdifferences;

	public SimpleProgressSummarizer getSummarizer(String file) throws Exception{
		return (SimpleProgressSummarizer)SimpleProgressSummarizer.load(null, file);		
	}
	
	public double[][] getTestStatsFromSummarizer(SimpleProgressSummarizer sps){
		return sps.getTrialAccuraciesTest();
	}
	
	//public int getBudget(SimpleProgressSummarizer spsA) throws Exception{
	//	double[][] statsA = getTestStatsFromSummarizer(spsA);
	//	return statsA.length;
	//}
	
	private void runKLvsRRstatplus() throws Exception {
		double[][][] algoStatsRR = new double[numIter][][];
		double[][][] algoStatsKL = new double[numIter][][];
		for(int d=0; d<numDataSets; d++)
			for(int i=0; i<numAlgos; i++){
				double sum = 0.0;
				for (int k=0; k<numIter; k++){
					algoStatsRR[k] = getTestStatsFromSummarizer(getSummarizer(serFiles_RR[d][i][k]));
					algoStatsKL[k] = getTestStatsFromSummarizer(getSummarizer(serFiles_KL[d][i][k]));
					for(int j=0; j<numEvalTrials; j++)
						sum += (Utils.mean(algoStatsRR[k][j]) - Utils.mean(algoStatsKL[k][j]))/numIter;
				}
				KLvsRRdifferences[d][i] = sum;
				//System.out.println("d="+d+",i="+i+sum);
			}
	}
	
	public void printKLvsRRLatexTable(){

		NumberFormat f = new DecimalFormat ( "#.##" );
		double[] temp_dbl = new double[numDataSets];

		//System.out.println("%%%%%%%%%%  Table: KL versus RR  %%%%%%%%%%");
		System.out.println("%%%%%%%%%%  Table:" +category+"   %%%%%%%%%%");
		System.out.println("\\begin{table*}               %");
		//System.out.println("\\caption{Area between the learning curves using Random Row Selection        %");
		//System.out.println("          and KL-Divergence.  A positive value indicates using KL-Divergence %");
		//System.out.println("          outperformed Random Row Selection.}                                %");
		System.out.println("\\caption{"+category.substring(0,2)+": Area between the learning curves using Row Selection "+category.substring(3,5)+"  %");
		System.out.println("          and "+category.substring(5,7)+".  A positive value indicates using "+category.substring(3,5)+"  %");
		System.out.println("          outperformed "+category.substring(5,7)+".}                                %");
		//System.out.println("\\label{table:KLvsRR} %");
		System.out.println("\\label{table:"+category+"} %");
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
				System.out.format(f.format(KLvsRRdifferences[d][i]));
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
				temp_dbl[d] = KLvsRRdifferences[d][i];
			System.out.format(f.format(Utils.mean(temp_dbl)));
		}
		System.out.print("  \\\\ \n");
		System.out.println("\\end{tabular}            %");
		System.out.println("\\end{table*}              %");
		System.out.println("\n\n");

	}
	
	public void printSummary(){
		
		System.out.println("Statistics Summary");
		System.out.println("=============================");
		System.out.println(" Budget:           "+Budget);
		System.out.println(" Trials Evaluated: "+numEvalTrials);
		System.out.println(" Datasets: ");
		for(int i=0; i<numDataSets; i++)
			System.out.println(" "+dataSets[i]);
		System.out.println("Algorithms: ");
		for(int i=0; i<numAlgos; i++)
			System.out.println(" "+algoNames[i]);
		System.out.println("\n\n");
		
		printKLvsRRLatexTable();
	}
	
	
	public static void main(String[] argv)throws Exception
	{
		RunKLvsRRStatsplus myStats = new RunKLvsRRStatsplus();
		ProjectEnv env= new ProjectEnv();
		env.put(ARGV, argv);
		myStats.setEnv(env);
		myStats.initialize();
		myStats.runKLvsRRstatplus();
		myStats.printSummary();		
	}

	public void initialize() throws IOException  {
		String[] argv= (String []) env.get(ARGV);
		
		
		try {
			if(argv.length==0||Utils.getFlag('h',argv)) {
				throw new Exception("Usage: RunStats -T budget "+
													"-category [\"category\""+
													"-numIter [\"Number of Iterations\""+
						                            "-datasets   [\"DataSet Names\"" + 
						                            "-algoNames  [\"Algorithm Names\"" +
						                            "-serFilesKL [\"list of serFilePrefixes (KL)\"]+" +
						                            "-serFilesRR [\"list of serFilePrefixes (RR)\"]\n");
				
			}
			
			category=Utils.getOption("category", argv);
						
			String current_numIter=Utils.getOption("numIter", argv);
			if (current_numIter.length()!=0)
				 numIter = Integer.parseInt(current_numIter);

			String listOfDataSets=Utils.getOption("datasets", argv);
			if (listOfDataSets.length()!=0)
				 dataSets = listOfDataSets.split("\\s+");
			
			String list_of_algoNames=Utils.getOption("algoNames", argv);
			if (list_of_algoNames.length()!=0)
				 algoNames = list_of_algoNames.split("\\s+");
			for(int i=0; i<algoNames.length; i++)
				System.out.println("DEBUG: algoNames["+i+"] = "+algoNames[i]);
			
		    String list_of_serFilesKL=Utils.getOption("serFilesKL", argv);
			if (list_of_serFilesKL.length()!=0)
				serFilePrefixesKL = list_of_serFilesKL.split("\\s+");
			for(int i=0; i<serFilePrefixesKL.length; i++)
				System.out.println("DEBUG: serFilePrefixesKL["+i+"] = "+serFilePrefixesKL[i]);

		    String list_of_serFilesRR=Utils.getOption("serFilesRR", argv);
			if (list_of_serFilesRR.length()!=0)
				serFilePrefixesRR = list_of_serFilesRR.split("\\s+");
			for(int i=0; i<serFilePrefixesRR.length; i++)
				System.out.println("DEBUG: serFilePrefixesRR["+i+"] = "+serFilePrefixesRR[i]);

			String temp_Budget = Utils.getOption("T", argv);
			if(temp_Budget.length()!=0)
				numEvalTrials = Integer.parseInt(temp_Budget);
			else
				numEvalTrials = -1;
				
		} catch (Exception e) {
			
			e.printStackTrace();
		}

		numDataSets = dataSets.length;
		numAlgos = algoNames.length;
		
		if(serFilePrefixesRR.length != numAlgos || serFilePrefixesKL.length != numAlgos) {
				System.err.println("Error: number of algorithms does not match number of serFiles");
			}
		

		/* This really sucks, there is not enough memory to load all the ser
		 * files, we'll have to do this on the fly. 
		 */
		
		serFiles_RR = new String[numDataSets][numAlgos][numIter];
		serFiles_KL = new String[numDataSets][numAlgos][numIter];

		for(int i=0; i<numDataSets; i++)
		  for(int j=0; j<numAlgos; j++)
			  for (int k=0; k<numIter; k++){
			  serFiles_RR[i][j][k] = serFileDirectory+(k+1)+"/"+serFilePrefixesRR[j]+"_"+dataSets[i]+".ser";
		      serFiles_KL[i][j][k] = serFileDirectory+(k+1)+"/"+serFilePrefixesKL[j]+"_"+dataSets[i]+".ser";
		      //System.out.println(serFiles_RR[i][j][k]);
		      //System.out.println(serFiles_KL[i][j][k]);
		  }

		try {
			Budget = numTrials;//getBudget(getSummarizer(serFiles_RR[0][0][0]));
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		if(numEvalTrials == -1)
			numEvalTrials = Budget;

		KLvsRRdifferences = new double[numDataSets][numAlgos];
		for(int d=0; d<numDataSets; d++)
			for(int i=0; i<numAlgos; i++)
				KLvsRRdifferences[d][i] = 0.0;		                                 
	}

	public void run() {
		// TODO Auto-generated method stub
		
	}
}
