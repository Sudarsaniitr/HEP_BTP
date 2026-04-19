#include <iostream>
#include <string>
#include <unordered_set> 
#include <unistd.h> 
#include <vector> 
#include<bits/stdc++.h>
#include <unistd.h>     // access, fork, execvp// waitpid
#include <cstdlib>      // getenv
#include <sstream> 
#include <sys/wait.h>   //   waitpid
#include <limits.h>
using namespace std;
unordered_set<string> commands = {"echo","type","exit","pwd","cd"};

bool is_builtin(const string& command) {
     if(commands.find(command)== commands.end())
     return false;
     return true;
}
void split_path(const string &path, vector<string> &dirs)
{
   string temp="";
  for(auto i : path)
  {
     if(i==';' || i==':')
     {
       dirs.push_back(temp);
       temp="";
       continue;
     }
     temp.push_back(i);
  }
}
vector<string> split_args(const string &s) {
    vector<string> result;
    istringstream iss(s);
    string token;
    while (iss >> token) {
        result.push_back(token);
    }
    return result;
}

bool is_executable(const string &path)
{
  return access(path.c_str(), X_OK)==0;
}
int main() {
  // Flush after every std::cout / std:cerr
  std::cout << std::unitbuf;
  std::cerr << std::unitbuf;

  // TODO: Uncomment the code below to pass the first stage
  while (true) {

   cout << "$ ";
 
   string command;
   cin>>command;
   string args;

   getline(cin, args);
   if(args.size()>0)
   args=args.substr(1);
   
   

   if(command == "exit") 
    break;

   bool f=false;


   
    if(is_builtin(command))
    {
      if(command=="pwd")
      {
          char cwd[PATH_MAX];
    if (getcwd(cwd, sizeof(cwd)) != NULL) {
        cout <<  cwd << endl;
        continue;
    } else {
        perror("getcwd() error");
        continue;
    }
        f=true;
        continue;
      }
      if(command=="echo")
      {
        cout<<args<<endl;
        f=true;
        continue;
      }
       if(command=="type")
      {
        string temp=args;
        if(is_builtin(args))
        {
        cout<<temp<<" " << "is a shell builtin"<<endl;
        continue;
        }
        else 
        {
          string path= getenv("PATH");
          vector<string> dirs;
          split_path(path,dirs);
          if(dirs.size()==0)
          {
          cout<<temp<<": " << "not found" << endl;
          continue;
          }
          else
          {
            for(auto it : dirs)
            {
              if(it.size()==0)
              continue;
              it+="/"+temp;
              if(is_executable(it))
              {
                cout<<temp<<" is "<<it<<endl;
                f=true;
                break;
              }
              else
              {
              continue;
              }

            }
            if(f)
            continue;
            else
            {
            cout<<temp<<": " << "not found" << endl;
            continue;
            }
          }


        }
        
        f=true;
      }

    if(f)
   continue;
     }
          else
     {
       // -------- Run external command using fork + execvp --------

       // Build argv strings: [command, arg1, arg2, ...]
       vector<string> argv_strings;
       argv_strings.push_back(command);

       vector<string> extra_args = split_args(args);
       argv_strings.insert(argv_strings.end(), extra_args.begin(), extra_args.end());

       // Convert to char* array for execvp
       vector<char*> argv;
       for (string &s : argv_strings) {
           argv.push_back(const_cast<char*>(s.c_str()));
       }
       argv.push_back(nullptr); // execvp expects NULL-terminated array

       pid_t pid = fork();

       if (pid < 0) {
           // fork failed
           perror("fork");
           // fall through to final "command not found" if you like, or just continue
           continue;
       }

       if (pid == 0) {
           // Child process: try to exec
           execvp(command.c_str(), argv.data());

           // If execvp returns, it failed
           cout << command << ": command not found" << endl;
           _exit(1);
       } else {
           // Parent process: wait for child to finish
           int status;
           waitpid(pid, &status, 0);
       }

       // We handled the command (or printed error in child),
       // so don't fall through to the final "command not found"
       continue;
     }

      
   
  
  

   cout<<command<<": " << "command not found" << endl;
  }
}