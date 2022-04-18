#include <stdio.h>
#include <string.h>
#include <ctype.h>
#include <stdlib.h>
#include <math.h>

int K = 0;
int K1 = 0;
int N = 0;
int max_iter =0;
char *input;
char *output;
int base = 10;
int i=0;
int j =0;
int i1=0;
int j1=0;
int z1=0;
int z=0;
int x =0;
int len=0;
char *ptr;
char *eptr;
FILE *inputFile = NULL;
FILE *outputFile = NULL;
int d=1;
char line[256];
double **clusters = 0;
double* centroids = 0;
double* centroidsHolder = 0;
double* dataPoints = 0;
char*token = NULL;
int iter =0;
double epsilon = 0.001;
double sum=0;
double norm=0;
double min=0;
int minIndex=0;
int temp =0;
int* numInCluster = 0;
int count = 0;
int a =0;
int b =0;
int c =0;
int rip =0;
int flag = 1;
char final[256];
int counter1=0;

int is_number(char *str, int *base)
{
    if (str == NULL) {
        return 0; }

    len = strlen(str);

    if (len == 1)
    {
        *base = 10;
        return isdigit(str[0]);
    }

    if ((str[0] == '0') && (str[1] == 'x'))
    {
        for (i = 2; i < len; i++)
        {
            char c = str[i];
            c = tolower(c);
            if (!(
                (c >= '0' && c <= '9') || 
                (c >= 'a' && c <= 'f'))) {
                return 0; }
        }
        *base = 16;
    }
    else
    {
        i = 0;
        if (str[0] == '-' || str[0] == '+') {
            i = 1; }

        for (; i < len; i++)
        {
            if (!isdigit(str[i])) {
                return 0; }
        }
        *base = 10;
    }
    return 1;
}

int main(int argc,char * argv[])
{
    if(is_number(argv[1],&base)==0) /* if k isn't an integer */
    {
        printf("Invalid Input!");
        return 1;
    } 
    if (argc != 4 && argc != 5) /* if number of inputs is invalid */
    {
        printf("Invalid Input!");
        return 1;
    }
    
    if (argc == 4) /* first case, without max_iter */
    {
        K = strtol(argv[1],&ptr,10);
        input = malloc( sizeof(char) * (strlen(argv[2])+1) );
        output = malloc( sizeof(char) * (strlen(argv[3])+1) );
        strcpy(input,argv[2]);
        strcpy(output,argv[3]);
        max_iter = 200; /* if not provided the default value is 200 */
    }
    else if(argc == 5) /* second case, with max_iter */
    {
        if(is_number(argv[2],&base)==0) /* checking if max_iter is an integer */
        {
            printf("Invalid Input!");
            return 1;
        } 
        K = strtol(argv[1],&ptr,10);
        max_iter= strtol(argv[2],&ptr,10);
        input = malloc( sizeof(char) * (strlen(argv[3])+1) );
        output = malloc( sizeof(char) * (strlen(argv[4])+1) );
        strcpy(input,argv[3]);
        strcpy(output,argv[4]);
    }
    if (max_iter<1)
    {
        printf("Invalid Input!");
        return 1;
    }


    inputFile = fopen(input,"r");

    if (inputFile == NULL) /* if file was not found */
    {
        printf("An Error Has Occurred");
        return 1;
    }
    
    fgets (line , 256 , inputFile);
    len = strlen(line);
    for (i=0;i<len;i++)
    {
        d += (line[i] == ','); /* calculating our d value */
    }   
    fclose(inputFile);

    inputFile = fopen(input,"r");
    
    while(fgets(line,256,inputFile)) /* calculating our N value */
    {
        line[strcspn(line,"\n")] =0; 
        N++;
    }
    fclose(inputFile);


    /* checking if K's input is valid:(1<K<N) */
    if (K<1 || K>=N)
    {
        printf("Invalid Input!\n");
        return 1;
    }

    dataPoints = realloc(dataPoints,d*N*sizeof(double));
    numInCluster= realloc(numInCluster,K*sizeof(int));
    centroids = realloc(centroids,d*K * sizeof(double));
    centroidsHolder=realloc(centroidsHolder,d*K*sizeof(double));
    for(i=0;i<K;i++)
    {
        clusters = (double **) realloc(clusters,K*sizeof(*clusters));
        clusters[i] = (double *)malloc(N*d*sizeof(double));
    }

    for (i=0;i<K;i++) /* init numInCluster */
    {
        numInCluster[i] = 0;
    }
    for (i=0;i<K;i++) /* init clusters */
    {
        for(j=0;j<N*d;j++)
        {
            clusters[i][j]=0;
        }
    }


    inputFile = fopen(input,"r");
    i=0;
    while(fgets(line,256,inputFile))
    {
        token = strtok(line,",");
        while (token != NULL) {
            dataPoints[i] = strtod(token,&eptr); /* initializing dataPoints */
            if (i<K*d) 
            {
                centroids[i] = dataPoints[i]; /* initializing centroids. */
            }
            i++;
            token = strtok(NULL, ",");
        }
    }
    fclose(inputFile);

    /* repeat: */
    while(flag==1 && iter<max_iter)
    {
        for (a=0;a<K;a++)
        {
            for(b=0;b<N*d;b++)
            {
                clusters[a][b]=0;
            }
        }
        for (x=0;x<K;x++) /*reset num in each cluster*/
        {
            numInCluster[x]=0;
        }
        for (i1=0; i1<N*d; i1=i1+d)
        {
            min = 10000000;
            for(j1=0;j1<K*d;j1=j1+d)
            {
                sum =0;
                for (z1=0;z1<d;z1++)
                {
                    sum = sum + pow(dataPoints[i1+z1]-centroids[j1+z1],2); /* (x_i-mu_j)^2*/
                }
                if (sum<min)
                {
                    min = sum;
                    minIndex = j1/d; /* d =/= 0 */
                }
            }
            for (z1=0;z1<d;z1++)
            {
                temp = numInCluster[minIndex];
                clusters[minIndex][temp] = dataPoints[i1+z1];
                numInCluster[minIndex]+=1;
            }
        }
        /* update the centroids */
        /* first we copy the current centroids to another array in order to check if new-old<epsilon) */
        for (c=0;c<K*d;c++)
        {
            centroidsHolder[c]=centroids[c];
        }
        /* calculate the new centroids*/
        rip=0;
        for (i1=0;i1<K;i1++)
        {   
            count=0;
            while(count<d) {
            sum=0;
            for(j1=count;j1<N*d;j1=j1+d)
            { 
                sum = sum + clusters[i1][j1];
            }
            centroids[rip+count] = sum/(numInCluster[i1]/d);
            count++;
            }
            rip = rip +d;
        }

        /* now we want to check if ||new-old||<epsilon for every vectors in centroids[] */
        for(c=0;c<K*d;c++)
        {
            centroidsHolder[c]= fabs(centroidsHolder[c])-fabs(centroids[c]);
        }
        

        counter1=0;
        for(c=0;c<K*d;c=c+d)
        {
            sum=0;
            for(z=0;z<d;z++)
            {
                sum = sum + pow(centroidsHolder[c+z],2);
            }
            norm = sqrt(sum);
            if (norm < epsilon) {
                counter1++;
                if (counter1==K)
                {
                    flag=0;
                }
            }
        }

        iter++;
    } 

    outputFile = fopen(output,"w"); /* insert the centroids to the output file */

    for(i=0;i<K*d;i=i+d)
    {
        for(j=0;j<d;j++)
        {
            snprintf(final,256,"%0.4f",centroids[i+j]);
            fputs(final,outputFile);
            if(j != d-1)
            {
                fputs(",",outputFile);
            }
        }
        fputc('\n',outputFile);
    }
    fclose(outputFile);


    free(numInCluster);
    free(dataPoints);
    for (i = 0; i < K; i++)
        free(clusters[i]);
    free(clusters);
    free(centroids);
    free(centroidsHolder);
    free(input);
    free(output); 
    return 0;
}