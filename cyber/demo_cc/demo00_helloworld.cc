
#include "cyber/cyber.h"

int main(int argc, char const *argv[])
{
    apollo::cyber::Init(argv[0]);
    AINFO << "hello world!";
    return 0;
}