package dk.blfw.alg;

import java.util.Arrays;
import java.util.Random;

public class Exp3{
	private static Random r=new Random();
	private Random random=null;
	private double gamma=0.05;
	private double[] w=null;
	private int K=0;
	private int lastT;
	Exp3(double a_gamma, int a_K){
		random= new Random(dk.blfw.Global.random.nextInt());
		gamma=a_gamma;
		K=a_K;
		lastT=-1;
		initWeights();
	}
	
	public void setLastT(int lastT) {
		this.lastT = lastT;
	}

	public int getLastT() {
		return lastT;
	}	
	
	public double[] getP(){
		double sum_w=0;
		for (int i=0; i<w.length; i++){
			sum_w+=w[i];
		}
		
		double[] p= new double[w.length];
		for (int i=0; i<w.length; i++){
			p[i]=(1-gamma)*w[i]/sum_w + gamma/K;
		}
		return p;
	}
	
	private void initWeights(){
		w= new double[K];
		for (int i=0; i<w.length;i++){
			w[i]=1;
		}
	
	}
	
	public static int rollDice(double probs[]){

        int dice = 0;
        double sum = 0.0;
        for(int i=0; i<probs.length; i++)
            sum += probs[i];
        if(sum != 1.0)
            System.err.println("ERROR: DiceRoll: probabilities do not sum to 1.0 (probs = " + sum);
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
	
	//TODO: refactor this code to use the above rollDice
	public int drawWeighted(){
		double[] P= getP();
		double[] cum_sum= new double[P.length];
		cum_sum[0]=P[0];
		for(int i=1; i<cum_sum.length; i++){
			cum_sum[i]=cum_sum[i-1]+P[i];
		}
		
		double key= dk.blfw.Global.random.nextDouble();
		int ret= Arrays.binarySearch(cum_sum, key);
		if (ret>=0){ return ret;
		}else{
			ret= -(ret+1);
			return ret;
		}
	}
	
	public int drawUniform(int num){
		return dk.blfw.Global.random.nextInt(num);
	}
	
	public void updateWithReward(double r){
		double[] x= new double[K];
		double[] p= getP();
		x[lastT]= r/ p[lastT];
		
		for(int i=0; i<w.length; i++){
			w[i]=w[i]*Math.exp(gamma*x[i]/K);
		}
	}


	
}
