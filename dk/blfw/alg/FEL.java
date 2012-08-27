package dk.blfw.alg;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;


public class FEL {
	private Random random = new Random();
	public enum ARGMAX {SUM, DIST_AVG, TIME_AVG};
	private static final double EPS_DEFAULT= 0.1;
	private static final int    M_DEFAULT = 5;
	double eps= EPS_DEFAULT;
	int    m = M_DEFAULT;
	List<double[]> s; 
	private double[]  sum;
	//private final TYPE type;
	private final ARGMAX argmax;
	
	// construct a FEL 
	public FEL(double eps, int n, ARGMAX argmax){
		this.eps = eps;
		this.s= new ArrayList<double[]>();
		this.sum = new double[n];
		this.argmax = argmax;
	}
	
	public FEL(double eps, int n){
		this(eps, n,ARGMAX.TIME_AVG);
	}
	
	public FEL(int n){
		this(EPS_DEFAULT, n);
	}
	
	public FEL shrink(int exclude){
		FEL copy= new FEL(eps, sum.length-1, argmax);
		
		for (int i=0,j=0; i< copy.sum.length; j++){
			if (j!=exclude){ 
				copy.sum[i]=sum[j]; i++;}
			                                
		}
		
		return copy;
	}
	
	public void incur(double[] cost){
		s.add(cost);
		for (int i = 0; i < cost.length; i++) {
			//handle case where we treat some of the costs unknown.
			if (argmax==ARGMAX.TIME_AVG) {
				if (Double.isNaN(cost[i])) continue;
			}
			
			sum[i]= cost[i]+sum[i];
		}
	}
	
	public void incur(int i, double c){
		double[] cost= new double[sum.length];
		
		if (argmax==ARGMAX.TIME_AVG) {
			for (int j=0; j<cost.length; j++)
				cost[j]= (j==i)? c: Double.NaN;
		}else {cost[i]=c;}
		
		incur(cost);
	}
	
	
	
	public int draw(){
		int tmp_i=-1;
		double tmp_value=Double.POSITIVE_INFINITY;
		double []p;
		
		
		switch (argmax) {
		case SUM:
			p= random(sum.length,1/eps);
			for(int i=0; i<sum.length; i++){
				if (sum[i]+p[i]<tmp_value){
					tmp_i=i;
					tmp_value= sum[i]+p[i];
				}
			}
			break;

		case DIST_AVG:
			int[] tmp= new int[m];
			for(int l=0; l< m; l++){
				p = random(sum.length,1/eps);
				for(int i=0; i<sum.length; i++){
					if ((sum[i]+p[i])<tmp_value){
						tmp[l]=i;
						tmp_value= sum[i]+p[i];
					}
				}
			}
			
			tmp_i = tmp[ dk.blfw.Global.random.nextInt(m)];
			break;
			
		case TIME_AVG:
			p= random(sum.length,1/(eps));
			
			for(int i=0; i<sum.length; i++){
				int c=0;
				for (int j=0; j< s.size();j++){
					if (! Double.isNaN( s.get(j)[i])) c++;
				}
				
				
				if ((sum[i]+p[i])/(c+1)<tmp_value){
					tmp_i=i;
					tmp_value= (sum[i]+p[i])/(c+1);
				}
			}
			break;			
			
		}
		
		return tmp_i;
		
	}
	
	
/*	@SuppressWarnings("unchecked")
	public int[] draw(){
		Integer[] rank= new Integer[sum.length];
		final double [] perturbed = new double[sum.length];
		double []p;
		
		
		switch (argmax) {
		case SUM:
			for (int i=0; i<rank.length; i++) rank[i]=i;
			p= random(sum.length,1/eps);
			for(int i=0; i<sum.length; i++){
				perturbed[i]= sum[i]+p[i];
			}
			
			Arrays.sort(rank, new Comparator(){
				public int compare(Object o1, Object o2) {
					// TODO Auto-generated method stub
					if (perturbed[(Integer)o1] < perturbed[(Integer)o2]) return -1;
					if (perturbed[(Integer)o1] > perturbed[(Integer)o2] ) return 1;
					return 0;
				}
			});
			
			int[] ret= new int[rank.length];
			for (int i=0; i<ret.length; i++) ret[i]=rank[i];
			return ret;
			break;
			

		case DIST_AVG:
			int[] tmp= new int[m];
			for(int l=0; l< m; l++){
				p = random(sum.length,1/eps);
				for(int i=0; i<sum.length; i++){
					if ((sum[i]+p[i])<tmp_value){
						tmp[l]=i;
						//tmp_value= sum[i]+p[i];
					}
				}
			}
			
			tmp_i = tmp[ random.nextInt(m)];
			break;
			
		case TIME_AVG:
			p= random(sum.length,1/(sum.length*eps));
			
			for(int i=0; i<sum.length; i++){
				int c=0;
				for (int j=0; j< s.size();j++){
					if (! Double.isNaN( s.get(i)[j])) c++;
				}
				
				
				if ((sum[i]/+p[i])/(c+1)<tmp_value){
					tmp_i=i;
					//tmp_value= sum[i]+p[i];
				}
			}
			break;			
			
		}
		
		return tmp_i;
		
	}
*/
	private double[] random(int n, double b){
		double[] a= new double[n];
		for(int i=0; i<a.length; i++){
			a[i]= dk.blfw.Global.random.nextDouble()*b;
		}
		return a;
	}

	
	public double[] getSum() {
		return sum;
	}
	

	
}
