#ifndef RECTCLASSSCORE_H_
#define RECTCLASSSCORE_H_

#include <sstream>
#include <string>

template<typename _Tp>
class RectClassScore
{
public:
  _Tp x, y, w, h;
  _Tp score;
  unsigned int classnum;
  bool enabled;

  inline std::string toString()
  {
    std::ostringstream out;
    // out << "P(" << GetClassString() << ") at " << "(x:" << x << ", y:" << y << ", w:" << w << ", h:" << h << ") ="
    //     << score;
    out << "P(" << GetClassString() << ") at " << "(x:" << x << ", y:" << y << ", w:" << w << ", h:" << h << ") ="
        << score;
    return out.str();
  }

  inline std::string GetClassString()
  {
    switch (classnum)
    {
      case 0:
        return "nothing";
      case 1:
        return "car";
      case 2:
        return "truck";
      case 3:
        return "bus";
      case 4:
        return "person";
      case 5:
        return "car_reg";
      case 6:
        return "car_big_reg";
      case 7:
        return "car_front";
      case 8:
        return "car_big_front";
      case 9:
        return "car_big_front";
      case 10:
        return "cow";
      case 11:
        return "table";
      case 12:
        return "dog";
      case 13:
        return "horse";
      case 14:
        return "motorbike";
      case 15:
        return "person";
      case 16:
        return "plant";
      case 17:
        return "sheep";
      case 18:
        return "sofa";
      case 19:
        return "train";
      case 20:
        return "tv";
      default:
        return "error";
    }
  }
};


#endif /* RECTCLASSSCORE_H_ */
