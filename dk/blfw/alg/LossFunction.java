package dk.blfw.alg;

import java.util.Enumeration;
import java.util.Random;

import weka.core.Instance;
import weka.core.Instances;

public class LossFunction{

    Random r;
    int N;

    public LossFunction(){
        r = new Random();
        N = 300;
    }

    public double GINI(int alpha[][][]){

    	//TODO: Debug, test
        /* Builds a model based on alpha_ijk
         * Randomly generates N fully specified, labeled instances based on this model
         * Computes the GINI index from this pool:
         * L_GINI = sum_y sum_x P(x)P(y|x)(1-P(y|x))
         * 
         * THIS FUNCTION HAS NOT BEEN DEBUGGED OR TESTED, DO NOT USE UNTIL WE ARE SURE IT WORKS
         */

        int numAttr   = alpha.length;
        int numLabels = alpha[0].length;

        int samples[][];
        int sample_labels[];
        samples = new int[N][];
        sample_labels = new int[N];
        for(int i=0; i<N; i++)
            samples[i] = new int[numLabels];


        //build the model
        double label_probs[];
        label_probs = new double[numLabels];
        //label_prob[j] = Pr(label = j):
        int sum_of_jk = 0;
        int sum_of_k = 0;
        for(int j=0; j<alpha[0].length; j++)
          for(int k=0; k<alpha[0][j].length; k++)
              sum_of_jk += alpha[0][j][k];

        for(int j=0; j<numLabels; j++)
        {
          sum_of_k = 0;
          for(int k=0; k<alpha[0][j].length; k++)
              sum_of_k += alpha[0][j][k];
          label_probs[j] = sum_of_k / sum_of_jk;
        }

        double attr_probs_given_j[][][];
        attr_probs_given_j = new double[numAttr][numLabels][];
        for(int i=0; i<numAttr; i++)
          for(int j=0; j<numLabels; j++)
            attr_probs_given_j[i][j] = new double[alpha[i][j].length];

        //Pr(a_i = q | label = y) = alpha_iyq / sum_k alpha_iyk:
        for(int i=0; i<numAttr; i++)
          for(int y=0; y<numLabels; y++)
            for(int k=0; k<attr_probs_given_j[i][y].length; k++)
            {
                sum_of_k = 0;
                for(int l=0; l<alpha[i][y].length; l++)
                    sum_of_k += alpha[i][y][l];
                attr_probs_given_j[i][y][k] = alpha[i][y][k] / sum_of_k;
            }


        //build samples
        double temp_double[];
        for(int x=0; x<N; x++)
        {
            sample_labels[x] = RollDice(label_probs);
            for(int i=0; i<numAttr; i++)
            {
                temp_double = attr_probs_given_j[i][sample_labels[x]];
                samples[x][i] = RollDice(temp_double);
            }
        }

        //calculate GINI index from samples
        double Loss = 0.0;
        double P_x = 0.0;
        double P_y_given_x = 0.0;
        //for(int j=0; j<numLabels; j++) we're switching the sums...
        for(int i=0; i<N; i++)
        {
              //calculate P(x) and P(y|x)
              int label_count[];
              label_count = new int[numLabels];
              int count=0;
              for(int l=0; l<N; l++)
              {
                  //if sample[i] = sample[l]
                  if(isEqual(samples, i, l))
                  {
                      count++;
                      label_count[sample_labels[l]]++;
                  }
              }
              P_x = count / N;
              for(int j=0; j<numLabels; j++)
              {
                  P_y_given_x = label_count[j] / count;
                  Loss += P_x * P_y_given_x * (1.0 - P_y_given_x);
              }

          }

        return Loss;
    }

    private boolean isEqual(int array[][], int index_1, int index_2)
    {
        boolean all_equal = true;
        for(int i=0; i<array[index_1].length; i++)
            if(array[index_1][i] != array[index_2][i])
                all_equal = false;
        return all_equal;

    }

    public double ConditionalEntropy(int alpha[][][], int attr_i){

        //alpha_ijk = # times an instance labeled j has attribute a_i = k
        // Loss_CE     = - sum_k( Pr(a_i = k) * sum_y( P(label = y | a_i = k) * log(P(label = y | a_i = k))))
        // Pr(a_i = k) = (sum_j alpha_ijk) / sum_j sum_k alpha_ijk
        // Pr(label = y | a_i = k) = alpha_iyk / sum_j alpha_ijk
        //System.out.println("DEBUG: ConditionalEntropy(alpha, i=" + attr_i + ")");
        //System.out.println("+++++++++++++++++++++++++++++++++++++++++++++");
        double Pr_A = 0.0, Pr_B = 0.0;

        double loss_sum = 0.0;

        int sum_of_jk = 0;
        for(int j=0; j<alpha[attr_i].length; j++)      //sum over all labels
          for(int k=0; k<alpha[attr_i][j].length; k++) //sum over all attribute values
            sum_of_jk += alpha[attr_i][j][k];
        //System.out.println("sum_of_jk = " + sum_of_jk);


        int sum_of_j = 0;
        double inner_sum = 0.0;
        for(int k=0; k<alpha[attr_i][0].length; k++)
        {
            sum_of_j = 0;
            for(int j=0; j<alpha[attr_i].length; j++)
                sum_of_j += alpha[attr_i][j][k];
            Pr_A = (double) sum_of_j / (double) sum_of_jk;
            //System.out.println("sum_of_j = " + sum_of_j + ", Pr_A = " + Pr_A);

            inner_sum = 0.0;
            for(int j=0; j<alpha[attr_i].length; j++)
            {
                Pr_B = (double) alpha[attr_i][j][k] / (double) sum_of_j;
                //System.out.println("Pr_B = " + Pr_B);
                inner_sum += Pr_B * Math.log(Pr_B) / Math.log(2.0);
            }

            loss_sum += Pr_A * inner_sum; // * sum_y;
        }
        //System.out.println("returned value = " + -1.0 * loss_sum);
        return -1.0 * loss_sum;
    }

    public double ConditionalEntropy(Instances pool, int attr_i){

        //initialize alpha's to one
    	int alpha[][][];
        int NumberOfFeatures = pool.numAttributes() - 1;
        int NumberOfLabels = pool.numClasses();

        alpha = new int[NumberOfFeatures][NumberOfLabels][];
        for(int i=0; i<NumberOfFeatures; i++)
          for(int j=0; j<NumberOfLabels; j++)
              alpha[i][j] = new int[pool.attribute(i).numValues()];

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
              Instance inst=e.nextElement();
              if(!inst.isMissing(i)) //if attribute i is not missing (i.e. its been bought)
              {
                  int j = (int) inst.classValue();
                  int k = (int) inst.value(i);
                  alpha[i][j][k]++;
              }
          }
        }
        return ConditionalEntropy(alpha, attr_i);

    }

    public double ExpectedClassificationError(int alpha[][][], int attr_i){

        //Expected Classification Error:
        //\argmin_{i} \sum{x} P (X_i = x) \min_{y} (1 - P (Y = y|Xi = x))
        //fix i, then
        //            \sum_k P(attr_i = k) * min_{y} [1 - P(label=y | attr_i = k)
        //where P(attr_i = k) = sum_j alpha[i][j][k] / sum_j,k alpha[i][j][k]
        //and   P(label=y | attr_i = k) =

        int sum_jk =0;
        for(int j=0; j<alpha[attr_i].length; j++)
          for(int k=0; k<alpha[attr_i][j].length; k++)
            sum_jk += alpha[attr_i][j][k];

        double P_a;
        double outer_sum = 0.0;
        for(int k=0; k<alpha[attr_i][0].length; k++)
        {
            int sum_j = 0;
            for(int j=0; j<alpha[attr_i].length; j++)
                sum_j += alpha[attr_i][j][k];
            P_a = (double) sum_j / (double) sum_jk;

            double min_y = 1.0 - ( (double) alpha[attr_i][0][k] / (double) sum_j );
            double temp = 0.0;
            for(int y=0; y<alpha[attr_i].length; y++)
            {
                temp = 1.0 - ( (double) alpha[attr_i][y][k] / (double) sum_j );
                if(min_y > temp)
                    min_y = temp;
            }
            outer_sum += P_a * min_y;
        }

        return outer_sum;
    }

    public double ExpectedClassificationError(Instances pool, int attr_i){

        //initialize alpha's to one
    	int alpha[][][];
        int NumberOfFeatures = pool.numAttributes() - 1;
        int NumberOfLabels = pool.numClasses();

        alpha = new int[NumberOfFeatures][NumberOfLabels][];
        for(int i=0; i<NumberOfFeatures; i++)
          for(int j=0; j<NumberOfLabels; j++)
              alpha[i][j] = new int[pool.attribute(i).numValues()];

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
              Instance inst=e.nextElement();
              if(!inst.isMissing(i)) //if attribute i is not missing (i.e. its been bought)
              {
                  int j = (int) inst.classValue();
                  int k = (int) inst.value(i);
                  alpha[i][j][k]++;
              }
          }
        }
        return ExpectedClassificationError(alpha, attr_i);

    }

    public int RollDice(double probs[]){

        int dice = 0;
        double sum = 0.0;
        for(int i=0; i<probs.length; i++)
            sum += probs[i];
//        if(sum != 1.0)
//            System.err.println("ERROR: DiceRoll: probabilities do not sum to 1.0 (probs = " + sum);
        double dice_prob = dk.blfw.Global.random.nextDouble();
        sum = 0.0;
        for(int i=0; i<probs.length; i++)
        {
            sum += probs[i];
            if(sum >= dice_prob)
            {
                dice = i;
                break;
            }
        }
        return dice;
    }

    public int RollDice(double probs[][]){

        int dice = 0;
        double sum = 0.0;
        for(int i=0; i<probs.length; i++)
          for(int j=0; j<probs[i].length; j++)
              sum += probs[i][j];
//        if(sum != 1.0)
//            System.err.println("ERROR: DiceRoll: probabilities do not sum to 1.0 (probs = " + sum);
        double dice_prob = dk.blfw.Global.random.nextDouble();
        sum = 0.0;
        for(int i=0; i<probs.length; i++)
          for(int j=0; j<probs[i].length; j++)
          {
            sum += probs[i][j];
            if(sum >= dice_prob)
            {
                dice = i;
                break;
            }
          }
        return dice;
    }

}

