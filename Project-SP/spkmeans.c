#include <stdio.h>
#include <string.h>
#include <ctype.h>
#include <stdlib.h>
#include <math.h>


char *file_name;
char *goal;
FILE *file = NULL;

int main(int argc,char * argv[])
{

    printf("--------------STARTC------------\n");
    if (argc != 3)
    {
        printf("Invalid Input!");
        return 1;
    }
    goal = malloc(sizeof(char)*(strlen(argv[1])+1));
    strcpy(goal,argv[1]);
    file_name = malloc(sizeof(char)*(strlen(argv[2])+1));
    strcpy(file_name,argv[2]);

    printf("%s,%s\n",goal,file_name);


    printf("--------------ENDC------------\n");
    return 0;
}