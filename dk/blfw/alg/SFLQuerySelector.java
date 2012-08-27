package dk.blfw.alg;

import java.util.EnumMap;
import java.util.Enumeration;
import java.util.Map;
import java.util.Random;

import weka.classifiers.Classifier;
import weka.core.Instance;
import weka.core.Instances;

import dk.blfw.core.IntfQuery;
import dk.blfw.core.QueryRequest;
import dk.blfw.impl.PhonyAttrQuery;
import dk.blfw.impl.QuerySelector;

public class SFLQuerySelector extends QuerySelector{

    /**
	 * 
	 */
	private static final long serialVersionUID = 1L;
	Random r = new Random();  //note to deng: Random(0) is bad, use Random() for time seed
    MyMaths OurMaths;
    LossFunction ExpectedLoss;
    RowSelector MyRowSelector;
    Instances pool;
    int NumberOfFeatures;
    int NumberOfLabels;
    int alpha[][][];
    boolean Initialized;
    boolean RANDOMIZED_SFL;
    public static final int CONDITIONAL_ENTROPY = 0;
    public static final int GINI_INDEX = 1;
    public static final int EXPECTED_CLASSIFICATION_ERROR = 2;
    int LOSS_FUNCTION_TYPE;
    boolean DEBUG;

    @SuppressWarnings("unchecked")

    public SFLQuerySelector(){
        //LOSS_FUNCTION_TYPE = CONDITIONAL_ENTROPY;
        LOSS_FUNCTION_TYPE = CONDITIONAL_ENTROPY;
        RANDOMIZED_SFL = true;
        Initialized = false;
        DEBUG = false;
    }

    public void SetLossFunctionType(int A){
        //question: can calling function use BLAH.SetLossFunctionType(GINI_INDEX)?
        if(A < 0 || A > 2)
          return;
        else
            LOSS_FUNCTION_TYPE = A;
    }

    public void SetRandomizedSFL(boolean A){
        RANDOMIZED_SFL = A;
    }

    private void InitializeParameters(){

        OurMaths = new MyMaths(pool.numInstances()+100); //assuming this is big enough
        MyRowSelector = new RowSelector();
        ExpectedLoss = new LossFunction();

        NumberOfFeatures = pool.numAttributes() - 1;  //leave out the class attribute
        NumberOfLabels = pool.numClasses();

        //allocate memory
        alpha = new int[NumberOfFeatures][NumberOfLabels][];
        for(int i=0; i<NumberOfFeatures; i++)
          for(int j=0; j<NumberOfLabels; j++)
              alpha[i][j] = new int[pool.attribute(i).numValues()];

        //initialize alpha's to one for the Dirichet distro
        for(int i=0; i<NumberOfFeatures; i++)
          for(int j=0; j<NumberOfLabels; j++)
            for(int k=0; k<alpha[i][j].length; k++)
                alpha[i][j][k] = 1;

        Initialized = true;
    }

    @Override
	@SuppressWarnings("unchecked")
	public IntfQuery propose(EnumMap<QueryRequest, Object> request, Map optional)
    {
        if(DEBUG){
            System.out.println("SFL Called");
            System.out.println("=============================");
        }

        pool = (Instances) request.get(QueryRequest.P);

        if(!Initialized)
            InitializeParameters();

        //make sure there should be at least one missing value somewhere.
        boolean hasmissing=false;
        for (Enumeration<Instance> e = pool.enumerateInstances(); e.hasMoreElements();){
            Instance i=e.nextElement();
            if (i.hasMissingValue()) {hasmissing=true; break;}
        }

        if (hasmissing==false){
            return IntfQuery.NONQUERY;
        }

        //SFL
        int argmin_i = 0, argmin_j = 0;
        double Prob_A = 0.0, Prob_B = 0.0;
        double Expectation = 1000000.0;
        int Sum_A = 0, Sum_B = 0;

        //initialize alpha's to one
        for(int i=0; i<NumberOfFeatures; i++)
          for(int j=0; j<NumberOfLabels; j++)
            for(int k=0; k<alpha[i][j].length; k++)
              alpha[i][j][k] = 1;

        //construct alpha's
        for(int i=0; i<NumberOfFeatures; i++)  //for each attribute
        {
          if(i == pool.classIndex())//skip the class attribute
              i++;
          for (Enumeration<Instance> e = pool.enumerateInstances(); e.hasMoreElements();) //for each instance
          {
              //alpha[i][pool.(instance j's label)][index of instance j's attribute i value]
              //wow, this is ugly
              Instance inst=e.nextElement();
              //inst.value(i) = instance inst, attribute a_i's value (double)
              if(!inst.isMissing(i)) //if attribute i is not missing (i.e. its been bought)
              {
                  //Deng claims that though attributes are floats internally,
                  //they are really of type int, hashed from nominal values
                  //  i = i-th attribute
                  //  j = current instance's class label (index associated)
                  int j = (int) inst.classValue();
                  //  k = current instance's i-th attribute value (index associated)
                  int k = (int) inst.value(i);
                  if( j != inst.classValue() || k != inst.value(i))
                  {
                    System.err.println("Deng was WRONG!!!");
                    System.err.println("integer j = " + j + " inst.classValue() = " + inst.classValue());
                    System.err.println("integer k = " + k + " inst.value(i=" + i + ") = " + inst.value(i));
                    System.err.println("NEVER TRUST DENG!");
                  }
                  //System.err.println("(i,j,k) = ("+i+","+j+","+k+")");
//                  if (i>=alpha.length || j>= alpha[i].length || k>= alpha[i][j].length){
//                	  System.err.println("(i,j,k) = ("+i+","+j+","+k+")");
//                        
//                  }
                  alpha[i][j][k]++;
                  //System.out.println("DEBUG: (constructing alpha's): i,j,k = "+ i + ", " + j + ", " + k + " = " + alpha);
              }
          }
        }

        if(false)
        {
          System.out.println("DEBUG: alpha[][][]:");
          for(int i=0; i<NumberOfFeatures; i++)
           for(int j=0; j<NumberOfLabels; j++)
            for(int k=0; k<alpha[i][j].length; k++)
                if(alpha[i][j][k] != 1)
                  System.out.println("alpha[" + i + "][" + j +"][" + k + "] = " + alpha[i][j][k]);
        }

        //deep copy alpha to alpha'
        int temp_int =0;
        int alpha_prime[][][];
        alpha_prime = new int[NumberOfFeatures][NumberOfLabels][];
        for(int i=0; i<NumberOfFeatures; i++)
          for(int j=0; j<NumberOfLabels; j++)
              alpha_prime[i][j] = new int[alpha[i][j].length];
        for(int i=0; i<NumberOfFeatures; i++)
          for(int j=0; j<NumberOfLabels; j++)
            for(int k=0; k<alpha[i][j].length; k++)
            {
                temp_int = alpha[i][j][k];
                alpha_prime[i][j][k] = temp_int;
            }

        double ExpectedLosses[][];
        ExpectedLosses = new double[NumberOfFeatures][NumberOfLabels];
        for(int i=0; i<NumberOfFeatures; i++)
          for(int j=0; j<NumberOfLabels; j++)
              ExpectedLosses[i][j] = 0.0;

        //start SFL
        boolean IsFeasible[][];
        IsFeasible = new boolean[NumberOfFeatures][NumberOfLabels];
        for(int i=0; i<NumberOfFeatures; i++)
          for(int j=0; j<NumberOfLabels; j++)
            IsFeasible[i][j] = false;
        boolean feasible = false;
        for(int i=0; i<NumberOfFeatures; i++)
        {
          if(i == pool.classIndex())//skip the class attribute
            i++;
          for(int j=0; j<NumberOfLabels; j++)
          {
            feasible = false;
            for(int I=0; I<pool.numInstances(); I++)
            {
                Instance inst = pool.instance(I);
                if( (int) inst.classValue() == j && inst.isMissing(i) )
                {
                    feasible = true;
                    IsFeasible[i][j] = true;
                }
            }

            if(feasible) //check to see if i, j is feasible
            {
              Expectation = 0.0;
              for(int K=0; K<alpha[i][j].length; K++)
              {
                  //create configuration s':
                  alpha_prime[i][j][K]++;

                  //sum for configuration s
                  Sum_A = 0;
                  for(int k=0; k<alpha[i][j].length; k++)
                      Sum_A += alpha[i][j][k];

                  //sum for configuration s'
                  Sum_B = 0;
                  for(int k=0; k<alpha_prime[i][j].length; k++)
                      Sum_B += alpha_prime[i][j][k];

                  Prob_A = OurMaths.GammaOverGamma(Sum_A, Sum_B);
                  Prob_B = 1.0;
                  for(int k=0; k<alpha[i][j].length; k++)
                  {
                      Prob_B = Prob_B * OurMaths.GammaOverGamma(alpha_prime[i][j][k], alpha[i][j][k]);
                      //System.out.println("DEBUG-----------------------------|");
                      //System.out.println(" K = "+K+", i,j,k = " + i+","+j+","+k);
                      //System.out.println("     alpha_prime[i][j][k] = " + alpha_prime[i][j][k]);
                      //System.out.println("           alpha[i][j][k] = " + alpha[i][j][k]);
                      //System.out.println("                   result = " + OurMaths.GammaOverGamma(alpha_prime[i][j][k], alpha[i][j][k]));
                  }
                  //System.out.println("DEBUG: Prob_B = " + Prob_B);
                  if(LOSS_FUNCTION_TYPE == CONDITIONAL_ENTROPY)
                    Expectation += Prob_A * Prob_B * ExpectedLoss.ConditionalEntropy(alpha_prime, i);
                  else if(LOSS_FUNCTION_TYPE == GINI_INDEX) //not operable yet?
                    Expectation += Prob_A * Prob_B * ExpectedLoss.GINI(alpha_prime);
                  else if(LOSS_FUNCTION_TYPE == EXPECTED_CLASSIFICATION_ERROR) //not operable yet
                      Expectation += Prob_A * Prob_B * ExpectedLoss.ExpectedClassificationError(alpha_prime, i);
                  else
                      System.out.println("ERROR: LOSS_FUNCTION_TYPE = "+LOSS_FUNCTION_TYPE+" not valid");

                  alpha_prime[i][j][K]--;
              }
              ExpectedLosses[i][j] = Expectation;
            }
          }
        }

        if(DEBUG)
        {
            System.out.println("DEBUG: Expectations:");
            for(int i=0; i<ExpectedLosses.length; i++)
            {
              for(int j=0; j<ExpectedLosses[i].length; j++)
                System.out.print(ExpectedLosses[i][j] + " ");
              System.out.println();
            }
            System.out.println("DEBUG: Feasibility:");
            for(int i=0; i<IsFeasible.length; i++)
            {
              for(int j=0; j<IsFeasible[i].length; j++)
                System.out.print(IsFeasible[i][j] + " ");
              System.out.println();
            }
        }

        boolean Gibbs = true;
        if(RANDOMIZED_SFL)
        {
          if(Gibbs)
          {
            double p[][];
            p = new double[NumberOfFeatures][NumberOfLabels];
            for(int i=0; i<NumberOfFeatures; i++)
              for(int j=0; j<NumberOfLabels; j++)
                  p[i][j] = 0.0;
            double sum_ij = 0.0;
            for(int i=0; i<NumberOfFeatures; i++)
                for(int j=0; j<NumberOfLabels; j++)
                	sum_ij += Math.exp(-1.0 * ExpectedLosses[i][j]);
            for(int i=0; i<NumberOfFeatures; i++)
                for(int j=0; j<NumberOfLabels; j++)
                {
                	if(IsFeasible[i][j])
                    	p[i][j] = Math.exp(-1.0 * ExpectedLosses[i][j]) / sum_ij;
                	else
                		p[i][j] = 0.0;
                }
            double temp_sum = 0.0;
            for(int i=0; i<NumberOfFeatures; i++)
                for(int j=0; j<NumberOfLabels; j++)
                	temp_sum += p[i][j];
            double dice_roll = dk.blfw.Global.random.nextDouble() * temp_sum;
            double sum = 0.0;
            boolean chosen = false;
            for(int i=0; i<p.length; i++)
                for(int j=0; j<p[i].length; j++)
                {
                    if(sum >= dice_roll && IsFeasible[i][j] && !chosen)
                    {
                        chosen = true;
                        argmin_i = i;
                        argmin_j = j;
                    }
                    sum += p[i][j];
                }
          }
          else
          {
            double sum = 0.0;
            /* This is correct, but fails: 
               for(int i=0; i<ExpectedLosses.length; i++)
                 for(int j=0; j<ExpectedLosses[i].length; j++)
                	ExpectedLosses[i][j] = 1.0 / ExpectedLosses[i][j]; 
            */
           for(int i=0; i<ExpectedLosses.length; i++)
              for(int j=0; j<ExpectedLosses[i].length; j++)
                  sum += ExpectedLosses[i][j];
            double dice_prob = dk.blfw.Global.random.nextDouble() * sum;
            sum = 0.0;
            boolean chosen = false;
            for(int i=0; i<ExpectedLosses.length; i++)
              for(int j=0; j<ExpectedLosses[i].length; j++)
              {
                sum += ExpectedLosses[i][j];
                if(sum >= dice_prob && IsFeasible[i][j] && !chosen)
                {
                    chosen = true;
                    argmin_i = i;
                    argmin_j = j;
                }
              }
        	
          }
        }
        else //not randomized, choose best (lowest) ExpectedLoss
        {
            double BestExpectation = Double.MAX_VALUE;
            for(int i=0; i<ExpectedLosses.length; i++)
              for(int j=0; j<ExpectedLosses[i].length; j++)
              {
                  if(BestExpectation > ExpectedLosses[i][j] && IsFeasible[i][j])
                  {
                      argmin_i = i;
                      argmin_j = j;
                      BestExpectation = ExpectedLosses[i][j];
                  }

              }
        }
        //buy the argmin_i-th attribute of an (the first) instance with label argmin_j;
        int attr = argmin_i; 
        int label = argmin_j;
		Classifier c = (Classifier) request.get(QueryRequest.C);
        int row = MyRowSelector.SelectRow_KLDivergence(pool, c, attr, label);
        if(row == -1)
            return IntfQuery.NONQUERY;
        else
            return new PhonyAttrQuery(row,attr);

        
    }//end IntfQuery



}

class MyMaths{

    int size;
    double LogGammas[];

    public MyMaths(){
        size = 100+1;
        LogGammas = new double[size];
        Initialize();
    }

    public MyMaths(int a){
        size = a+1;
        LogGammas = new double[size];
        Initialize();
    }

    private void Initialize(){

        double Logs[] = new double[size];
        Logs[0] = 0.0;
        for(int i=1; i<size; i++)
            Logs[i] = Math.log(i);

        for(int i=0; i<LogGammas.length; i++)
          LogGammas[i] = 0.0;

        for(int i=1; i<LogGammas.length; i++)
        {
            //compute log( Gamma(i) ) = log(1) + log(2) + .. + log(i-1)
            for(int j=1; j<=i-1; j++)
                LogGammas[i] += Logs[j];
        }
    }

    public double GammaOverGamma(int a, int b){
        //Computes Gamma(a)/Gamma(b) = (a-1)!/(b-1)!
        //but uses logs so that its not too big

        //first check that a, b are in range
        if(a <= 0 || b <= 0)
        {
            System.err.println("ERROR in GammaOverGamma: a = " + a + ", b = " + b);
        }
        if(a+1 > size || b+1 > size)
        {
            System.err.println("ERROR in GammaOverGamma: a = " + a + ", b = " + b);
        }

        //log( Gamma(a) / Gamma(b)) = log(Gamma(a)) - log(Gamma(b))
        double GoG = LogGammas[a] - LogGammas[b];
        return Math.exp(GoG);

    }
}