{
  description = "CSCE 896: NLP Project";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs";
    utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, utils }:
    utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs {
          inherit system;
          config = {
            allowUnfree = true;
          };
        };
      in
      rec {
        packages = rec {

          edit_nts = pkgs.python37.pkgs.buildPythonPackage {
            pname = "edit_nts";
            version = "0.0.1";

            src = ./.;

            propagatedBuildInputs = with pkgs.python37.pkgs ; [
              pandas
              matplotlib
              numpy
              tqdm
              tensorflow
              tensorflow-tensorboard
              tensorflowWithCuda
              tensorflow-estimator
              Keras
              keras-preprocessing
              spacy
              scipy
              scikitlearn
            ];
          };

          pythonEnv = pkgs.python37.withPackages (ps: with ps; [
            edit_nts
            pandas
            matplotlib
            numpy
            tqdm
            tensorflow
            tensorflow-tensorboard
            tensorflowWithCuda
            tensorflow-estimator
            Keras
            keras-preprocessing
            spacy
            scipy
            scikitlearn
          ]);

          myEnv = pkgs.mkShell {
            buildInputs = [
              packages.pythonEnv
              pkgs.pre-commit
            ];
          };
        };

        defaultPackage = packages.pythonEnv;
        devShell = packages.myEnv;
      }
    );
}
