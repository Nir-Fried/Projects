static int wam(int K,int d,int N,int observations[],double dataPoints[],char* goal);
static int ddg(int K,int d,int N,int observations[],double dataPoints[],char* goal,double ** W);
static int lnorm(int K,int d,int N,int observations[],double dataPoints[],char* goal,double ** W,double ** D);
static int jacobi(int K,int d,int N,int observations[],double dataPoints[],char* goal,double ** lNorm);
static int eigengap(int K,int d,int N,int observations[],double eigenvalues[],double ** V);
static int spk(int K,int d,int N,int observations[],double ** V);

