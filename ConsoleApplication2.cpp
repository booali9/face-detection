#include <iostream>
#include <fstream>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/objdetect.hpp>
#include <vector>
#include <unordered_map>
#include <memory>
#include <ctime>

using namespace std;
using namespace cv;

const string databasePath = "C:/project/Database/Database "; // Path to store user data
const string attendancePath = "C:/project/Database"; // Path to store attendance data

// Base class for any person in the system
class Person {
public:
    Person(const string& name, int id) : name(name), id(id) {}
    virtual ~Person() = default;

    virtual void display() const {
        cout << "Name: " << name << ", ID: " << id << endl;
    }

    int getId() const {
        return id;
    }

    string getName() const {
        return name;
    }

    virtual string getDepartment() const { return ""; }
    virtual string getSubject() const { return ""; }

protected:
    string name;
    int id;
};

// Derived class for Student
class Student : public Person {
public:
    Student(const string& name, int id, const string& department)
        : Person(name, id), department(department) {}

    void display() const override {
        Person::display();
        cout << "Department: " << department << endl;
    }

    string getDepartment() const override {
        return department;
    }

private:
    string department;
};

// Derived class for Teacher
class Teacher : public Person {
public:
    Teacher(const string& name, int id, const string& subject)
        : Person(name, id), subject(subject) {}

    void display() const override {
        Person::display();
        cout << "Subject: " << subject << endl;
    }

    string getSubject() const override {
        return subject;
    }

private:
    string subject;
};

// Interface for face recognition
class FaceRecognizer {
public:
    virtual ~FaceRecognizer() = default;
    virtual bool recognize(const Mat& frame, int& id) = 0;
    virtual void registerFace(int id, const Mat& face) = 0;
};

// Simple implementation of FaceRecognizer using OpenCV
class SimpleFaceRecognizer : public FaceRecognizer {
public:
    SimpleFaceRecognizer(const string& modelPath) {
        if (!faceCascade.load(modelPath)) {
            cerr << "Failed to load face cascade model from path: " << modelPath << endl;
            throw runtime_error("Failed to load face cascade model.");
        }
    }

    bool recognize(const Mat& frame, int& id) override {
        vector<Rect> faces;
        faceCascade.detectMultiScale(frame, faces, 1.1, 3, 0, Size(30, 30));
        if (faces.empty()) {
            return false;
        }

        // Draw green rectangles around detected faces
        for (const auto& faceRect : faces) {
            rectangle(frame, faceRect, Scalar(0, 255, 0), 2); // Green color
        }

        // Extract the face region (considering only the first detected face)
        Mat face = frame(faces[0]);

        // Simulate recognition by comparing with registered faces
        for (const auto& entry : faceDatabase) {
            if (compareFaces(entry.second, face)) {
                id = entry.first;
                return true;
            }
        }
        return false;
    }

    void registerFace(int id, const Mat& face) override {
        faceDatabase[id] = face.clone();

        // Save the face image to file
        string filename = "C:/project/Database/Database" + to_string(id) + ".jpg"; // Path to store picture
        imwrite(filename, face);
    }

private:
    bool compareFaces(const Mat& face1, const Mat& face2) {
        // Simple comparison: check if the size and pixel values are similar
        if (face1.size() != face2.size()) return false;
        double diff = norm(face1, face2);
        return diff < 1000; // Adjust threshold as needed
    }

    CascadeClassifier faceCascade;
    unordered_map<int, Mat> faceDatabase;
};

// Class for handling timestamps
class Time {
public:
    Time() {}

    string getCurrentTimestamp() const {
        time_t now = time(0);
        struct tm timeinfo;
        localtime_s(&timeinfo, &now);
        char buffer[26];
        strftime(buffer, sizeof(buffer), "%Y-%m-%d %H:%M:%S", &timeinfo);
        return string(buffer);
    }

    void displaytime(const string& timestamp) const {
        cout << "--------------------------" << endl;
        cout << "Timestamp: " << timestamp << endl;
        cout << "--------------------------" << endl;
    }
};

// Manages the overall attendance system
class AttendanceSystem {
public:
    AttendanceSystem(unique_ptr<FaceRecognizer> recognizer)
        : recognizer(move(recognizer)), currentFrame() {}

    const Mat& getCurrentFrame() const {
        return currentFrame;
    }

    void markAttendance() {
        VideoCapture cap(0);
        if (!cap.isOpened()) {
            cerr << "Error: Could not open camera" << endl;
            return;
        }

        Mat frame;
        while (true) {
            cap >> frame;
            if (frame.empty()) {
                cerr << "Error: Captured empty frame!" << endl;
                break;
            }

            currentFrame = frame.clone();  // Store the current frame
            Mat grayFrame;
            cvtColor(frame, grayFrame, COLOR_BGR2GRAY);

            int id;
            if (recognizer->recognize(grayFrame, id)) {
                if (idMap.find(id) != idMap.end()) {
                    cout << "Marking attendance for:" << endl;
                    idMap[id]->display();
                    logAttendance(idMap[id]);
                }
                else {
                    cout << "Unknown person detected. Registering new person..." << endl;
                    registerNewPerson(currentFrame);  // Call function to register new person
                }
            }

            imshow("Mark Attendance", frame);
            if (waitKey(10) == 27) break; // Exit on 'ESC' key press
        }

        cap.release();
        destroyAllWindows();
    }

    void registerNewPerson(const Mat& frame) {
        int id;
        string name, department, subject;

        cout << "Enter ID: ";
        cin >> id;
        cin.ignore(); // Ignore remaining newline character

        cout << "Enter Name: ";
        getline(cin, name);

        cout << "Enter Department: ";
        getline(cin, department);

        cout << "Enter Subject: ";
        getline(cin, subject);

        // Register as either Student or Teacher
        char role;
        cout << "Is the person a Student (S) or Teacher (T)? ";
        cin >> role;
        cin.ignore(); // Ignore remaining newline character

        if (tolower(role) == 's') {
            addPerson(make_shared<Student>(name, id, department));
        }
        else if (tolower(role) == 't') {
            addPerson(make_shared<Teacher>(name, id, subject));
        }
        else {
            cerr << "Invalid role. Please enter 'S' for Student or 'T' for Teacher." << endl;
            return;
        }

        // Register the face for the new person
        recognizer->registerFace(id, frame);

        cout << "New person registered and attendance marked." << endl;
    }

private:
    void addPerson(shared_ptr<Person> person) {
        people.push_back(person);
        idMap[person->getId()] = person;
        savePersonDetails(person);
    }

    void savePersonDetails(shared_ptr<Person> person) {
        string filename = databasePath + "person_details.txt";
        ofstream outFile(filename, ios::app);
        if (!outFile) {
            cerr << "Failed to open person details file: " << filename << endl;
            return;
        }

        outFile << "ID: " << person->getId() << ", Name: " << person->getName();
        if (dynamic_pointer_cast<Student>(person)) {
            outFile << ", Department: " << dynamic_pointer_cast<Student>(person)->getDepartment() << endl;
        }
        else if (dynamic_pointer_cast<Teacher>(person)) {
            outFile << ", Subject: " << dynamic_pointer_cast<Teacher>(person)->getSubject() << endl;
        }
        else {
            outFile << endl;
        }
        outFile.close();

        cout << "Person details saved for: " << person->getName() << endl;
    }

    void logAttendance(shared_ptr<Person> person) {
        string filename = attendancePath + "attendance.txt"; // File path for attendance log

        ofstream outFile(filename, ios::app); // Open file in append mode
        if (!outFile) {
            cerr << "Failed to open attendance log file: " << filename << endl;
            return;
        }

        Time time;
        string timestamp = time.getCurrentTimestamp();

        outFile << "Time: " << timestamp << ", ID: " << person->getId() << ", Name: " << person->getName();
        if (dynamic_pointer_cast<Student>(person)) {
            outFile << ", Department: " << dynamic_pointer_cast<Student>(person)->getDepartment() << endl;
        }
        else if (dynamic_pointer_cast<Teacher>(person)) {
            outFile << ", Subject: " << dynamic_pointer_cast<Teacher>(person)->getSubject() << endl;
        }
        else {
            outFile << endl;
        }
        outFile.close();

        cout << "Attendance logged for: " << person->getName() << endl;
    }

    vector<shared_ptr<Person>> people;
    unordered_map<int, shared_ptr<Person>> idMap;
    unique_ptr<FaceRecognizer> recognizer;
    Mat currentFrame;  // Store the current frame for registering new person
};

// Main function
int main() {
    string modelPath = "C:/cascade/haarcascade_frontalface_default.xml"; // Path to frontal face cascade

    try {
        unique_ptr<FaceRecognizer> recognizer = make_unique<SimpleFaceRecognizer>(modelPath);
        AttendanceSystem system(move(recognizer));

        cout << "Attendance system initialized." << endl;

        int choice;
        while (true) {
            cout << "1. Mark Attendance\n2. Register New Person\n3. Exit\nChoose an option: ";
            cin >> choice;
            cin.ignore(); // Ignore remaining newline character

            if (choice == 1) {
                system.markAttendance();
            }
            else if (choice == 2) {
                cout << "Registering new person..." << endl;

                VideoCapture cap(0);
                if (!cap.isOpened()) {
                    cerr << "Error: Could not open camera" << endl;
                    continue;
                }

                Mat frame;
                cap >> frame;
                if (!frame.empty()) {
                    Time time;
                    string timestamp = time.getCurrentTimestamp();
                    time.displaytime(timestamp);

                    system.registerNewPerson(frame);
                }
                else {
                    cerr << "Error: No frame captured to register new person." << endl;
                }
                cap.release();
            }
            else if (choice == 3) {
                break; // Exit the loop
            }
            else {
                cout << "Invalid option. Try again." << endl;
            }
        }
    }
    catch (const exception& e) {
        cerr << "Exception caught: " << e.what() << endl;
    }
    catch (...) {
        cerr << "Unknown exception caught." << endl;
    }

    return 0;
}