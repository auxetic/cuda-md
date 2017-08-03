#ifndef system_h
#define system_h

#define ratio   1.4
#define Pi      3.1415926535897932
#define sysdim  2

// type define
typedef struct
    {
    int     natom;
    double  phi;
    double  x, y;
    double  xinv, yinv;
    double  strain;
    } tpbox;

typedef struct
    {
    int     seed;
    double  phi;
    } tpsets;

// variables define
static tpbox    box;
static tpsets   sets;

#endif
