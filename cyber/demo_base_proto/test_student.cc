/*
 * 操作student对象，实现内容的读写
 * 流程：
 * 1. 包含头文件
 * 2. 创建student对象
 * 3. 写数据
 * 4. 读数据 
*/

#include "cyber/demo_base_proto/student.pb.h"
#include <iostream>
using namespace std;
int main(int argc, char const *argv[])
{
    apollo::cyber::demo_base_proto::Student stu;
    stu.set_name("huluwa");
    stu.set_age(7);
    stu.set_height(1.75);
    stu.add_books("yuwen");
    stu.add_books("shuxue");
    stu.add_books("yingyu");

    string name = stu.name();
    uint64_t age = stu.age();
    double height = stu.height();
    
    cout << "name:" << name << "; age:" << age << "; height: " << height << endl;
    for(int i=0;i<stu.books_size();i++)
    {
        string book = stu.books(i);
        cout << book << "-";
    }
    cout << endl;
    return 0;
}
