/**
 * Copyright (c) 2015, Jozef Stefan Institute, first version done by Andrej Muhic, Jan Rupnik, 
 * final refactoring, upgrade of the idea and templatization done by Andrej Muhic
 * All rights reserved.
 * 
 * This source code is licensed under the FreeBSD license found in the
 * LICENSE file in the root directory of this source tree.
 */

#ifndef CROSSLINGUAL_H
#define CROSSLINGUAL_H

#include "base.h"
#include "mine.h"

namespace TCrossLingual {

	
	template<typename TSizeTy>
	using TIndexVec = TVec < TNum<TSizeTy>, TSizeTy >;

	template<typename TSizeTy, typename val>
	using TSparseVec = TPair < TIndexVec<TSizeTy>, TVec<TNum<val>, TSizeTy> > ;

	template<typename TSizeTy, typename val>
	using TSparseMatrix = TTriple < TIndexVec<TSizeTy>, TIndexVec<TSizeTy>, TVec<TNum<val>, TSizeTy > > ;

	///////////////////////////////
	// TCLProjectors
	//   Interface to projector classes
	template <class val = double, class TSizeTy = int, bool colmajor = false>
	class TCLProjectorAbstract {
	public:
		// project a sparse document	
		// What about storage, does it make sense to templatize this!
		virtual void Project(const TPair<TVec<TNum<TSizeTy>, TSizeTy>, TVec<TNum<val>, TSizeTy>> & Doc, const TStr& DocLangId, const TStrPr& TargetSpace, TVec<TNum<val>, TSizeTy>& Projected, const TBool& DmozSpecial = true) const = 0;
		// project a sparse matrix (columns = documents)
		virtual void Project(const TTriple<TVec<TNum<TSizeTy>, TSizeTy>, TVec<TNum<TSizeTy>, TSizeTy>, TVec<TNum<val>, TSizeTy>>& DocMatrix, const TStr& DocLangId,  const TStrPr& TargetSpace, TVVec<TNum<val>, TSizeTy, colmajor>& Projected, const TBool& Tfidf = true, const TBool& DmozSpecial = true) const = 0;
		//get reference to proxy matrix
		//TODO: add Project
		virtual bool GetProxyMatrix(const TStr& DocLangId, const TStr& Lang2Id, const TVVec<TNum<val>, TSizeTy, colmajor>*& Proxy, bool& transpose_flag, const TStr& Friend) const = 0;
		virtual bool GetCenters(const TStr& DocLangId, const TStr& Lang2Id, const TVec<TNum<val>, TSizeTy> *& Center1, const TVec<TNum<val>, TSizeTy>*& Center2, const TStr& Friend) const = 0;
		virtual ~TCLProjectorAbstract(){};
	};
	typedef TCLProjectorAbstract<> TCLProjector;

	///////////////////////////////
	// TCLHubProjector
	//   All languages are aligned with a common (hub) language, typically English
	template <class val = double, class TSizeTy = int, bool colmajor = false>
	class TCLHubProjectorAbstract : public TCLProjectorAbstract<val, TSizeTy, colmajor> {
	private:
		// Projectors: cols = num words, rows = num projectors
		TVec< TPair< TVVec<TNum<val>, TSizeTy, colmajor>, TVVec<TNum<val>, TSizeTy, colmajor> > > Projectors;
		// centroid length = num words
		TVec< TPair<TVec<TNum<val>, TSizeTy>, TVec<TNum<val>, TSizeTy> > > Centers;
		// inverse document frequency vectors
		TVec< TPair< TVec<TNum<val>, TSizeTy>, TVec<TNum<val>, TSizeTy> > > InvDoc;
		// Pairs of language ids
		THash<TPair<TStr, TStr>, TInt> LangIds;
		// Connection matrices for non-direct comparissons: sim(x,y) = x' X C_{X,Y} Y' y, based on two connections to the hub language: (H_X, X), (H_Y, Y)
		TVec< TPair<TVVec<TNum<val>, TSizeTy, colmajor>, TVVec<TNum<val>, TSizeTy, colmajor> > > ConMatrices;
		// maps pairs of lang ids to an index in ConMatrices
		THash<TPair<TStr, TStr>, TInt> ConIdxH;
		// the index used to determine which hub basis to choose from Projectors when comparing documents in the hub language (example: when
		// comparing two English documents with Projectors{en_de, en_fr}, there are two choices
		int HubHubProjectorIdx;
		TStr Friend;
		// Hub language ID (example: "en")
		TStr HubLangId;
		// Maps language identifiers to indices in Projectors, Centers, ConMatrices
		// DocLangId -> idx1,  Lang2Id  -> idx2 in Projectors and centers and index in ConMatrices
		bool SelectMatrices(const TStr& DocLangId, const TStr& Lang2Id, const TVVec<TNum<val>, TSizeTy, colmajor>*& ProjMat, const TVec<TNum<val>, TSizeTy>*& Center, const TVVec<TNum<val>, TSizeTy, colmajor>*& ConMat, const TVec<TNum<val>, TSizeTy>*& InvDocV, const TBool& DmozSpecial = true) const;
		bool SelectMatrices(const TStr& DocLangId, const TPair<TStr, TStr>& TargetSpace, const TVVec<TNum<val>, TSizeTy, colmajor>*& ProjMat, const TVec<TNum<val>, TSizeTy>*& Center, const TVVec<TNum<val>, TSizeTy, colmajor>*& ConMat, const TVec<TNum<val>, TSizeTy>*& InvDocV, TBool& transpose) const;

	public:
		TCLHubProjectorAbstract() { HubHubProjectorIdx = 0; }
		// Load binary matrices
		void Load(const TStr& ModelFolder, const TStr& HubHubProjectorDir);
		// Load binary matrices use config files
		void Load(const TStr& ProjectorPathsFNm, const TStr& ConMatPathsFNm, const TStr& HubHubProjectorLangPairId, const bool& onlyhub = true);

		void LoadHub(const TStr& ProjectorPath, const TStr& ConMatrixPath, const TStrPr& TargetSpace){
			Projectors.Gen(1, 1); Centers.Gen(1, 1); InvDoc.Gen(1, 1); LangIds.AddDat(TargetSpace, 0); ConMatrices.Gen(1, 1);
			HubHubProjectorIdx = 0; ConIdxH.AddDat(TargetSpace, 0); Friend = TargetSpace.Val2; HubLangId = TargetSpace.Val1;
			TStr Name = HubLangId + "_" + Friend;
			TStr Path = ProjectorPath + "/" + Name + "/";
			TPair<TVVec<TNum<val>, TSizeTy, colmajor>, TVVec<TNum<val>, TSizeTy, colmajor>>& ProjectorsPair = Projectors[0];
			TPair<TVec<TNum<val>, TSizeTy>, TVec<TNum<val>, TSizeTy>>&   CentersPair = Centers[0];
			TPair<TVec<TNum<val>, TSizeTy>, TVec<TNum<val>, TSizeTy>>&   InvDocPair = InvDoc[0];

			TFIn Center1File(Path + "/c1.bin");
			//TFltV c1(Center1File);
			CentersPair.Val1.Load(Center1File);
			TFIn Center2File(Path + "/c2.bin");
			//TFltV c2(Center2File);
			CentersPair.Val2.Load(Center2File);
			//Centers.Add(TPair<TFltV,TFltV>(c1, c2));
			///Centers.Add(CentersPair);
			TFIn idoc1In(Path + "/invdoc1.bin");
			//TFltV idoc1(idoc1In);
			InvDocPair.Val1.Load(idoc1In);
			TFIn idoc2In(Path + "/invdoc2.bin");
			//TFltV idoc2(idoc2In);
			InvDocPair.Val2.Load(idoc2In);
			//InvDoc.Add(TPair<TFltV, TFltV>(idoc1, idoc2));
			//InvDoc.Add(InvDocPair);
			TFIn Projector1File(Path + "/P1.bin");
			//TFltVV P1(Projector1File);
			//Andrej: we need only projector for a friend
			ProjectorsPair.Val1.Load(Projector1File);
			TFIn Projector2File(Path + "/P2.bin");
			//TFltVV P2(Projector2File);
			ProjectorsPair.Val2.Load(Projector2File);

			Path = ConMatrixPath + "/" + Name + ".bin";
			TFIn File(Path);
			TPair<TVVec<TNum<val>, TSizeTy, colmajor>, TVVec<TNum<val>, TSizeTy, colmajor>>&   ConMat = ConMatrices[0];
			ConMat.Load(File);
			printf("One Done\n");

		}

		void Project(const TPair<TVec<TNum<TSizeTy>, TSizeTy>, TVec<TNum<val>, TSizeTy>>& Doc, const TStr& DocLangId, const TStrPr& TargetSpace, TVec<TNum<val>, TSizeTy>& Projected, const TBool& DmozSpecial = true) const override;

		void Project(const TTriple<TVec<TNum<TSizeTy>, TSizeTy>, TVec<TNum<TSizeTy>, TSizeTy>, TVec<TNum<val>, TSizeTy>>& DocMatrix, const TStr& DocLangId, const TStrPr& TargetSpace, TVVec<TNum<val>, TSizeTy, colmajor>& Projected, const TBool& Tfidf = true, const TBool& DmozSpecial = true) const override;

		void DoTfidf(TSparseMatrix<val, TSizeTy>& DocMatrix, const TStr& DocLangId, const TStr& Lang2Id);
		void DoTfidf(TSparseVec<val, TSizeTy>& Doc, const TStr& DocLangId, const TStr& Lang2Id);
		bool GetProxyMatrix(const TStr& DocLangId, const TStr& Lang2Id, const TFltVV*& Proxy, bool& transpose_flag, const TStr& Friend) const;
		bool GetCenters(const TStr& DocLangId, const TStr& Lang2Id, const TVec<TNum<val>, TSizeTy>*& Center1, const TVec<TNum<val>, TSizeTy>*& Center2, const TStr& Friend) const override;
		bool GetCenter(const TStr& DocLangId, const TPair<TStr, TStr>& TargetSpace, const TVec<TNum<val>, TSizeTy>*& Center) const;
		void SetFriend(TStr& _Friend){ this->Friend = _Friend; }
	};

	typedef TCLHubProjectorAbstract<> TCLHubProjector;
#include "crosslingual_HubProj.inc"
	///////////////////////////////
	// TCLPairwiseProjector
	//   Language pairs have different common spaces. Includes logic for comparing documents in the same language
	class TCLPairwiseProjector : public TCLProjector {
	private:
		// Projectors: cols = num words, rows = num projectors
		TVec<TPair<TFltVV, TFltVV> > Projectors;
		// centroid length = num words
		TVec<TPair<TFltV, TFltV> > Centers;
		// Pairs of language ids
		TVec<TPair<TStr, TStr> > LangIds;
	public:
		TCLPairwiseProjector() {}
		// Load binary matrices
		void Load(const TStr& ModelFolder);
		// project a sparse document	
		void Project(const TPair<TIntV, TFltV>& Doc, const TStr& DocLangId, const TStrPr& TargetSpace, TFltV& Projected, const TBool& DmozSpecial = true) const {}
		// project a sparse matrix (columns = documents)
		void Project(const TTriple<TIntV, TIntV, TFltV>& DocMatrix, const TStr& DocLangId, const TStrPr& TargetSpace, TFltVV& Projected, const TBool& Tfidf = true, const TBool& DmozSpecial = true) const {}
		bool GetProxyMatrix(const TStr& DocLangId, const TStr& Lang2Id, const TFltVV*& Proxy, bool& transpose_flag, const TStr& Friend) const { return false; }
		bool GetCenters(const TStr& DocLangId, const TStr& Lang2Id, const TFltV*& Center1, const TFltV*& Center2, const TStr& Friend) const = 0;
	};


	///////////////////////////////
	// TCLCommonSpaceProjector
	//   Projects to a common vector space
	class TCLCommonSpaceProjector : public TCLProjector {
	private:
		// Projectors[i]: cols = num words, rows = num projectors for the i-th language
		TVec<TFltVV> Projectors;
		// length(Centers[i]) = num words in language i
		TVec<TFltV> Centers;
		TVec<TStr> LangIds;
	public:
		TCLCommonSpaceProjector() {}
		// project a sparse document	
		void Project(const TPair<TIntV, TFltV>& Doc, const TStr& DocLangId, const TStrPr& TargetSpace, TFltV& Projected, const TBool& DmozSpecial = true) const {}
		// project a sparse matrix (columns = documents)
		void Project(const TTriple<TIntV, TIntV, TFltV>& DocMatrix, const TStr& DocLangId, const TStrPr& TargetSpace, TFltVV& Projected, const TBool& Tfidf = true, const TBool& DmozSpecial = true) const {}
		bool GetProxyMatrix(const TStr& DocLangId, const TStr& Lang2Id, const TFltVV*& Proxy, bool& transpose_flag, const TStr& Friend) const { return false; }
		bool GetCenters(const TStr& DocLangId, const TStr& Lang2Id, const TFltV*& Center1, const TFltV*& Center2, const TStr& Friend) const = 0;
	};


	///////////////////////////////
	// TCLCore
	//   Core cross-lingual class that provides: vector space representation, projections and similarity computation
	template <class val=double, class TSizeTy = int, bool colmajor = false>
	class TCLCoreAbstract {
	private:
		TCLProjectorAbstract<val, TSizeTy, colmajor>* Projectors;
		THash<TStr, TUnicodeVSM::PGlibUBow> Tokenizers;
		TVec<TStr> LangIdV; // i-th language <==> i-th unicode tokenizer
		TVec<TIntV> InvDocFq;
		TVec<THash<TStr, TInt> > BowMap; // each language gets a hash map that maps tokenized input words to indices
		TStr Friend;

	public:
		TCLCoreAbstract(TCLProjectorAbstract<val, TSizeTy, colmajor>* Projectors_, const TStrV& TokenzierLangIdV, const TStrV& TokenizerPaths);
		TCLCoreAbstract(TCLProjectorAbstract<val, TSizeTy, colmajor>* Projectors_, const TStr& TokenizerPathsFNm);
		//TO-DO
		//void GetProxyMatrix(TStr Lang1, TStr Lang2){
		//	Projectors->
		//}
		// Text to sparse vector
		void TextToVector(TUStr& Text, const TStr& LangId, TSparseVec<TSizeTy, val> & SparseVec);
		// Vector of Unicode Text documents to DocumentMatrix
		void TextToVector(TUStrV& Docs, const TStr& LangId, TSparseMatrix<TSizeTy, val>& DocMatrix);
		void Project(TUStr& Text, const TStr& DocLangId, const TStrPr& TargetSpace, TVec<TNum<val>, TSizeTy>& Projected, const TBool& DmozSpecial);
		void Project(const TSparseVec<TSizeTy, val>& Doc, const TStr& DocLangId, const TStrPr& TargetSpace, TVec<TNum<val>, TSizeTy>& Projected, const TBool& DmozSpecial);
		// project vector of textual documents	
		void Project(TUStrV& Text, const TStr& DocLangId, const TStrPr& TargetSpace, TVVec<TNum<val>, TSizeTy, colmajor>& Projected, const TBool& DmozSpecial = true);
		// project a sparse document	
		void Project(const TSparseMatrix<TSizeTy, val>& DocMatrix, const TStr& DocLangId, const TStrPr& TargetSpace, TVVec<TNum<val>, TSizeTy, colmajor>& Projected, const TBool& DmozSpecial = true);
		// Cosine similarity, text input
		double GetSimilarity(TUStr& Text1, TUStr& Text2, const TStr& Lang1Id, const TStr& Lang2Id);
		// Cosine similarity, sparse vector input
		double GetSimilarity(const TSparseVec<TSizeTy, val>& Doc1, const TSparseVec<TSizeTy, val>& Doc2, const TStr& Lang1Id, const TStr& Lang2Id);
		// Cosine similarity, sparse document matrix input
		void GetSimilarity(const TSparseMatrix<TSizeTy, val>& DocMtx1, const TSparseMatrix<TSizeTy, val>& DocMtx2, const TStr& Lang1Id, const TStr& Lang2Id, TVVec<TNum<val>, TSizeTy, colmajor>& SimMtx);
		// Returns top k (weight, wordid) pairs in lang1 and lang2 that contributed most to the similarity
		double ExplainSimilarity(const TSparseVec<TSizeTy, val>& Doc1, const TSparseVec<TSizeTy, val>& Doc2, const TStr& Lang1Id, const TStr& Lang2Id, TVec<TPair<TNum<val>, TNum<TSizeTy>>, TSizeTy>& Lang1Words, TVec<TPair<TNum<val>, TNum<TSizeTy>>, TSizeTy>& Lang2Words);
		// Returns top k words as TUStr in lang1 and lang2 that contributed most to the similarity
		double ExplainSimilarity(TUStr& Text1, TUStr& Text2, const TStr& Lang1Id, const TStr& Lang2Id, TUStrV& Lang1TopWords, TUStrV& Lang2TopWords, const int& k);
		// Return word string (unicode bow)
		TUStr GetWordByKeyIdLangId(const TStr& LangId, const TNum<TSizeTy>& WordIndex) { return Tokenizers.GetDat(LangId)->GetWord(WordIndex); }
		// GetSimilarity for two vectors with given language Ids, proxy matrix is not applied, be careful with connection matrices select!!!!! is not symmetric
		// GetSimilarity for two matrices with given language Ids, proxy matrix is not applied, be careful with connection matrices select!!!!! is not symmetric
		// GetSimilarity for vector and matrix with given language Ids, proxy matrix is not applied, be careful with connection matrices select!!!!! is not symmetric
		void SetFriend(TStr& _Friend){ this->Friend = _Friend; }


	};
	typedef TCLCoreAbstract<> TCLCore;

	/*
	class TCLCore {
	private:
		TCLProjector* Projectors;
		THash<TStr, TUnicodeVSM::PGlibUBow> Tokenizers;
		TVec<TStr> LangIdV; // i-th language <==> i-th unicode tokenizer
		TVec<TIntV> InvDocFq;
		TVec<THash<TStr, TInt> > BowMap; // each language gets a hash map that maps tokenized input words to indices
		TStr Friend;

	public:
		TCLCore(TCLProjector* Projectors_, const TStrV& TokenzierLangIdV, const TStrV& TokenizerPaths);
		TCLCore(TCLProjector* Projectors_, const TStr& TokenizerPathsFNm);
		//TO-DO
		//void GetProxyMatrix(TStr Lang1, TStr Lang2){
		//	Projectors->
		//}
		// Text to sparse vector
		void TextToVector(TUStr& Text, const TStr& LangId, TPair<TIntV, TFltV>& SparseVec);
		// Vector of Unicode Text documents to DocumentMatrix
		void TextToVector(TUStrV& Docs, const TStr& LangId, TTriple<TIntV, TIntV, TFltV>& DocMatrix);
		// project a textual document	
		void Project(TUStr& Text, const TStr& DocLangId, const TStrPr& TargetSpace, TFltV& Projected, const TBool& DmozSpecial = true);
		// project vector of textual documents	
		void Project(TUStrV& Text, const TStr& DocLangId, const TStr& Lang2Id, TFltVV& Projected, const TBool& DmozSpecial = true);
		// project a sparse document	
		// project a sparse document	
		void Project(const TPair<TIntV, TFltV>& Doc, const TStr& DocLangId, const TStrPr& TargetSpace, TFltV& Projected, const TBool& DmozSpecial = true);
		// project a sparse matrix (columns = documents)
		void Project(const TTriple<TIntV, TIntV, TFltV>& DocMatrix, const TStr& DocLangId, const TStr& Lang2Id, TFltVV& Projected, const TBool& DmozSpecial = true);
		// Cosine similarity, text input
		double GetSimilarity(TUStr& Text1, TUStr& Text2, const TStr& Lang1Id, const TStr& Lang2Id);
		// Cosine similarity, sparse vector input
		double GetSimilarity(const TPair<TIntV, TFltV>& Doc1, const TPair<TIntV, TFltV>& Doc2, const TStr& Lang1Id, const TStr& Lang2Id);
		// Cosine similarity, sparse document matrix input
		void GetSimilarity(const TTriple<TIntV, TIntV, TFltV>& DocMtx1, const TTriple<TIntV, TIntV, TFltV>& DocMtx2, const TStr& Lang1Id, const TStr& Lang2Id, TFltVV& SimMtx);
		// Returns top k (weight, wordid) pairs in lang1 and lang2 that contributed most to the similarity
		double ExplainSimilarity(const TPair<TIntV, TFltV>& Doc1, const TPair<TIntV, TFltV>& Doc2, const TStr& Lang1Id, const TStr& Lang2Id, TVec<TPair<TFlt, TInt> >& Lang1Words, TVec<TPair<TFlt, TInt> >& Lang2Words);
		// Returns top k words as TUStr in lang1 and lang2 that contributed most to the similarity
		double ExplainSimilarity(TUStr& Text1, TUStr& Text2, const TStr& Lang1Id, const TStr& Lang2Id, TUStrV& Lang1TopWords, TUStrV& Lang2TopWords, const int& k);
		// Return word string (unicode bow)
		TUStr GetWordByKeyIdLangId(const TStr& LangId, const TInt& WordIndex) { return Tokenizers.GetDat(LangId)->GetWord(WordIndex); }
		// GetSimilarity for two vectors with given language Ids, proxy matrix is not applied, be careful with connection matrices select!!!!! is not symmetric
		// GetSimilarity for two matrices with given language Ids, proxy matrix is not applied, be careful with connection matrices select!!!!! is not symmetric
		// GetSimilarity for vector and matrix with given language Ids, proxy matrix is not applied, be careful with connection matrices select!!!!! is not symmetric
		void SetFriend(TStr& _Friend){ this->Friend = _Friend; }


	};*/

	//typedef TCLCoreAbstract<> TCLCore;

	///////////////////////////////
	// TCLDmozNode
	//   Dmoz classifier node
	template <class val = double, class TSizeTy = int, bool colmajor = false>
	class TCLDmozNodeAbstract {
	public:
		TCLDmozNodeAbstract() {}
		void Disp();
		void Save(TSOut& SOut) const;
		explicit TCLDmozNodeAbstract(TSIn& SIn) : DocIdxV(SIn), CatStr(SIn), CatPathStr(SIn), StrToChildIdH(SIn){}
		friend class TCLDmoz;
	private:
		TIntV DocIdxV; //DocIdxV = indices of documents for a given node	
		TStr CatStr;
		TStr CatPathStr;
		TStrIntH StrToChildIdH; // string -> child Id		
	};
	typedef TCLDmozNodeAbstract<> TCLDmozNode;

	template <class val = double, class TSizeTy = int, bool colmajor = false>
	class TCLClassifierAbstract;
	///////////////////////////////
	// TCLCNode
	//   Dmoz classifier node
	template <class val = double, class TSizeTy = int, bool colmajor = false>
	class TCLCNodeAbstract {
	public:
		TCLCNodeAbstract() {}
		void Disp();
		void Save(TSOut& SOut) const;
		explicit TCLCNodeAbstract(TSIn& SIn) : DocIdxV(SIn), CatStr(SIn), CatPathStr(SIn), StrToChildIdH(SIn) { }
		void Load(TSIn& SIn) { DocIdxV.Load(SIn); CatStr.Load(SIn); CatPathStr.Load(SIn); StrToChildIdH.Load(SIn); }
		friend class TCLClassifierAbstract<val, TSizeTy, colmajor>;
	private:
		TIntV DocIdxV; //DocIdxV = indices of documents for a given node	
		TStr CatStr; // Category id
		TStr CatPathStr; // Category path
		TStrIntH StrToChildIdH; // string -> child Id	
	};


	///////////////////////////////
	// TCLClassifier
	//   Dmoz classifier
	//class TCLClassifier;
	//typedef TPt<TCLClassifier> PCLClassifier;
	template <class val, class TSizeTy, bool colmajor>
	class TCLClassifierAbstract {
	private:
		TCRef CRef;
	public:
		friend class TPt<TCLClassifierAbstract>;
	private:
		// Core cross-lingual functionalities
		TCLCore* CLCore;
		TStr DmozPath;
		TStr ModelPath; //path where the model is stored

		TVec<TStr> Cat;
		//TVec<TVec<TIntKd> > Doc;	
		TTriple<TIntV, TIntV, TFltV> CatDocs;

		TStr HierarchyLangId; // dmoz -> "en"
		TStr Lang2Id;
		TStr TargetSpace;

		//TTree is not templatized properly, val->TNum<val>
		TTree<TCLCNodeAbstract< val, TSizeTy, colmajor> > Taxonomy;
		TFltVV PCatDocs;
		TFltVV Centroids;

		int CutDepth;

	public:
		TCLClassifierAbstract() : CutDepth(0), CLCore(nullptr) {};
		static TPt<TCLClassifierAbstract> New() { return TPt<TCLClassifierAbstract>(new TCLClassifierAbstract()); }
		TCLClassifierAbstract(const TStr& DmozPath_, const TStr& ModelPath_, const TStr& HierarchyLangId_, const TStr& Lang2Id_, TCLCore* CLCore_, int CutDepth_ = -1) { DmozPath = DmozPath_; ModelPath = ModelPath_; HierarchyLangId = HierarchyLangId_; Lang2Id = Lang2Id_; CLCore = CLCore_; CutDepth = CutDepth_; }
		static TPt<TCLClassifierAbstract> New(const TStr& DmozPath_, const TStr& ModelPath_, const TStr& HierarchyLangId_, const TStr& Lang2Id_, TCLCore* CLCore_, int CutDepth_ = -1) { return TPt<TCLClassifierAbstract>(new TCLClassifierAbstract(DmozPath_, ModelPath_, HierarchyLangId_, Lang2Id_, CLCore_, CutDepth_)); }
		// Loads the data
		void LoadData();
		// Load model
		void LoadModel();
		// Projects dmoz documents
		void ProjectDmoz();
		// Computes the centroids	
		void ComputeModel();
		// Get the centroid specific category string
		TStr& GetClassStr(const int& NodeId) { return Taxonomy.GetNodeVal(NodeId).CatStr; }
		// Get the centroid specific category path string
		TStr& GetClassPathStr(const int& NodeId) { return Taxonomy.GetNodeVal(NodeId).CatPathStr; }// }
		// Classify a single document
		void Classify(TUStr& Text, TInt& Class, const TStr& TextLangId);
		// Classify a single document, return top k classes
		void Classify(TUStr& Text, TIntV& Class, const TStr& TextLangId, const int& k = 10, const TBool& DmozSpecial = true);
		// Classify projected test document
		void ClassifyProjected(const TFltV& ProjVec, TInt& Class);
		// Classify projected test document, return top k classes
		void ClassifyProjected(const TFltV& ProjVec, TIntV& Class, const int& k);

		// Classify projected tests document, return top k classes
		void ClassifyProjected(const TFltVV& ProjVecVV, TVec<TIntV> & ClassVV, const int& k);
		// Get most frequent keywords from class path vector
		void GetBestKWordV(const TStrV& PathV, TStrV& KeywordV, TIntV& KeyFqV);
		// Get most frequent keywords from class path vector
		void GetBestKWordV(const TIntV& ClassV, TStrV& KeywordV, TIntV& KeyFqV);
		// Get most frequent keywords from class path vector
		void GetBestKWordV(const TIntV& ClassV, TStrV& KeywordV, TIntV& KeyFqV, TStrV& Categories);
		// Get most frequent keywords from class path vector
		void GetBestKWordV(const TVec<TIntV> & ClassVV, TVec<TStrV>& KeywordVV, TVec<TIntV>& KeyFqV, TVec<TStrV>& Categories);
	private:
		//// Building and using the model
		// Tree breadth first search
		//template <class val>
		void TreeBFS(const TTree<TCLCNodeAbstract<val, TSizeTy, colmajor> >& Tree, TIntV& IdxV) {
			IdxV.Gen(Tree.GetNodes(), 0);
			TQQueue<TInt> Queue;
			Queue.Push(0); //root
			while (!Queue.Empty()) {
				int NodeId = Queue.Top();
				IdxV.Add(NodeId);
				Queue.Pop();
				for (int ChildN = 0; ChildN < Tree.GetChildren(NodeId); ChildN++) {
					Queue.Push(Tree.GetChildNodeId(NodeId, ChildN));
				}
			}
		}
		////// Loading, saving, displaying, stats 
		// Save tree ID strings into a file based on a given ordering of elements
		void PrintTreeBFS(TTree<TCLCNodeAbstract<val, TSizeTy, colmajor>>& Tree, const TIntV& IdxV, const TStr& FNm);

		//// Loads an int vector (ascii)
		//void LoadTIntV(const TStr& FilePath, TIntV& Vec);
		//// Saves an int vector (ascii)
		//void SaveTIntV(const TStr& FilePath, TIntV& Vec);
		//// Loads a string vector (ascii)
		//void LoadTStrV(const TStr& FilePath, TStrV& Vec);
		//// Saves document class paths (matlab sparse matrix - use load -ascii and spconvert)
		//void SaveDocumentClasses();
		//// Saves centroid class paths (matlab sparse matrix - use load -ascii and spconvert)
		//void SaveCentroidClasses();
		//// Saves path strings to a file
		//void SaveCentroidClassesPaths();		
		//// Computes and saves taxonomy statistics (node depths and number of documents per node)
		//void TreeStats();						
	};

	typedef TCLClassifierAbstract<> TCLClassifier;
	typedef TPt<TCLClassifier> PCLClassifier;



	///////////////////////////////
	// TCLStream
	//   Comparing streams of documents
	class TCLStream {
	private:
		// Core cross-lingual functionalities
		TCLCore* CLCore;
		TVec<TStr> StreamLangIds; // language of each stream
		TIntV ProjStreamIdxV; // language index (points to StreamLangIds)
		TVec<TPair<TStr, TStr> > ProjStreamLangPairs; // sorted language pairs, autogenerated by using StreamLangIds and TrackPairs
		//**********************************************************************************************************************

		//// 4 languages: en, es, de, fr	
		//// 5 streams with their indices
		// 0 <-> en_bloomberg
		// 1 <-> en_news
		// 2 <-> es_news
		// 3 <-> de_news
		// 4 <-> fr_news

		//StreamLangIds[0] = en;
		//StreamLangIds[1] = en;
		//StreamLangIds[2] = es;
		//StreamLangIds[3] = de;
		//StreamLangIds[4] = fr;

		//// Selection of pairs of streams to track
		// TrackPairs[0] = 0, 1 ; track similarities in SimMatrices[0] // compare en_bloomberg and en_news
		// TrackPairs[1] = 1, 2 ; track similarities in SimMatrices[1] // compare en_news and es_news
		// TrackPairs[2] = 0, 3 ; track similarities in SimMatrices[2] // compare en_bloomberg and de_news
		// TrackPairs[3] = 2, 4 ; track similarities in SimMatrices[3] // compare es_news and fr_news
		// TrackPairs[4] = 0, 2 ; track similarities in SimMatrices[4] // compare en_bloomberg and es_news

		//**********************************************************************************************************************

		//// CCA (Pairwise) projectors (three pairs)
		// Projectors[0] = Pen, Pes; LangIds[0] = en, es
		// Projectors[1] = Pen, Pde; LangIds[1] = en, de
		// Projectors[2] = Pes, Pfr; LangIds[2] = es, fr

		// en_bloomberg <-> en_news
		// ProjStreams[0] = en_bloomberg projected with Projectors, ProjStreamLangPairs lang1 = en, lang2 = en
		// ProjStreams[1] = en_news projected with Projectors, ProjStreamLangPairs lang1 = en, lang2 = en
		// en_news <-> es_news
		// ProjStreams[2] = en_news projected with Projectors, ProjStreamLangPairs lang1 = en, lang2 = es
		// ProjStreams[3] = es_news projected with Projectors, ProjStreamLangPairs lang1 = es, lang2 = en
		// en_bloomberg <->  de_news
		// ProjStreams[4] = en_bloomberg projected with Projectors, ProjStreamLangPairs lang1 = en, lang2 = de
		// ProjStreams[5] = de_news projected with Projectors, ProjStreamLangPairs lang1 = de, lang2 = en
		// es_news <-> fr_news
		// ProjStreams[6] = es_news projected with Projectors, ProjStreamLangPairs lang1 = es, lang2 = fr
		// ProjStreams[7] = fr_news projected with Projectors, ProjStreamLangPairs lang1 = fr, lang2 = es
		// en_bloomberg <-> es_news
		// ProjStreams[8] = en_bloomberg projected with Projectors, ProjStreamLangPairs lang1 = en, lang2 = es
		// We do not need ProjStreams[9] = ProjStreams[3] = es_news projected with Projectors, ProjStreamLangPairs lang1 = es, lang2 = en

		// ProjStreamIdxV[0] = 0 // en_bloomberg
		// ProjStreamIdxV[1] = 1 // en_news
		// ProjStreamIdxV[2] = 1 // en_news
		// ProjStreamIdxV[3] = 2 // es_news
		// ProjStreamIdxV[4] = 0 // en_bloomberg
		// ProjStreamIdxV[5] = 3 // de_news
		// ProjStreamIdxV[6] = 2 // es_news
		// ProjStreamIdxV[7] = 4 // fr_news
		// ProjStreamIdxV[8] = 0 // en_bloomberg

		// SimMatrices[0] = ProjStreams[0]'*ProjStreams[1] // en_bloomberg <-> en_news
		// SimMatrices[1] = ProjStreams[2]'*ProjStreams[3] // en_news <-> es_news
		// SimMatrices[2] = ProjStreams[4]'*ProjStreams[5] // en_bloomberg <->  de_news
		// SimMatrices[3] = ProjStreams[6]'*ProjStreams[7] // es_news <-> fr_news
		// SimMatrices[4] = ProjStreams[8]'*ProjStreams[3] // en_bloomberg <-> es_news

		//**********************************************************************************************************************

		//// MCCA (Common) projectors
		// Projectors[0] = Pen; LangIds[0] = en
		// Projectors[1] = Pes; LangIds[1] = es
		// Projectors[2] = Pde; LangIds[2] = de
		// Projectors[3] = Pfr; LangIds[3] = fr

		// ProjStreams[0] = en_bloomberg projected with Projectors, ProjStreamLangPairs lang1 = en, lang2 = ignored
		// ProjStreams[1] = en_news projected with Projectors, ProjStreamLangPairs lang1 = en, lang2 = ignored
		// ProjStreams[2] = es_news projected with Projectors, ProjStreamLangPairs lang1 = es, lang2 = ignored
		// ProjStreams[3] = de_news projected with Projectors, ProjStreamLangPairs lang1 = de, lang2 = ignored
		// ProjStreams[4] = fr_news projected with Projectors, ProjStreamLangPairs lang1 = fr, lang2 = ignored

		// ProjStreamIdxV[0] = 0 // en_bloomberg
		// ProjStreamIdxV[1] = 1 // en_news
		// ProjStreamIdxV[2] = 2 // es_news
		// ProjStreamIdxV[3] = 3 // de_news
		// ProjStreamIdxV[4] = 4 // fr_news

		// SimMatrices[0] = ProjStreams[0]'*ProjStreams[0] // en_bloomberg <-> en_news
		// SimMatrices[1] = ProjStreams[0]'*ProjStreams[1] // en_news <-> es_news
		// SimMatrices[2] = ProjStreams[0]'*ProjStreams[2] // en_bloomberg <->  de_news
		// SimMatrices[3] = ProjStreams[1]'*ProjStreams[3] // es_news <-> fr_news
		// SimMatrices[4] = ProjStreams[0]'*ProjStreams[1] // en_bloomberg <-> es_news

		//**********************************************************************************************************************

		// Each matrix corresponds to particular projection of some stream, the columns are the newest projected documents
		TVec<TFltVV> ProjStreams;
		// Column index of the newest document in each matrix in ProjStreams;
		TVec<TInt> LatestDocIdx;


		// Pairs of indices of ProjStreams that are to be compared
		TVec<TPair<TInt, TInt> > TrackPairs;
		// Similarity matrices between pairs in TrackPairs
		TVec<TFltVV> SimMatrices;
	public:
		TCLStream(const TIntV& BufferSizes, const TIntV& ProjRowSizes, const TVec<TStr>& StreamLangIds_, const TVec<TPair<TInt, TInt> >& TrackPairs_) : StreamLangIds(StreamLangIds_), TrackPairs(TrackPairs_) {
			// TODO: Set LatestDocIdx = 0
			// TODO: Gen ProjStreams and ProjStreamLangPairs (using StreamLangIds and TrackPairs)
			// TODO: Gen SimMatrices		
		}
		// Add document given text and its language id 
		void AddDocument(TUStr& Text, const TInt& StreamIdx) {
			TIntV ProjStreamIdxV;
			// TODO: Find all ProjStreams indices where ProjStreamIdxV == StreamIdx and add them to StreamIdxV		
			TStr LangId = StreamLangIds[StreamIdx];
			// Compute vector space model (fill SparseVec) for LangId outside of the loop
			TPair<TIntV, TFltV> SparseVec;
			CLCore->TextToVector(Text, LangId, SparseVec);
			for (int ProjStreamN = 0; ProjStreamN < ProjStreamIdxV.Len(); ProjStreamN++) {
				AddDocument(SparseVec, ProjStreamIdxV[ProjStreamN]);
			}
			// TODO: Compute all required similarities (based on TrackPairs and LatestDocIdx)
		}
		// Add document given text and stream index
		void AddDocument(const TPair<TIntV, TFltV>& SparseVec, const TInt& ProjStreamIdx) {
			// Use CLCore + ProjStreamLangPairs[ProjStreamIdx] to update the StreamIdx matrix				
			TFltV ProjText;
			CLCore->Project(SparseVec, ProjStreamLangPairs[ProjStreamIdx].Val1, TStrPr(ProjStreamLangPairs[ProjStreamIdx].Val1, ProjStreamLangPairs[ProjStreamIdx].Val2), ProjText, false);
			AddDocument(ProjText, ProjStreamIdx);
		}
		// Add projected document in a given stream
		void AddDocument(const TFltV& Doc, const TInt& ProjStreamIdx) {
			// TODO: Update ProjStreams[ProjStreamIdx] matrix
			// TODO: Increment LatestDocIdx[ProjStreamIdx]		
		}
		// TODO: Getters

	};


	inline void TCLPairwiseProjector::Load(const TStr& ModelFolder_) {
		// Important:
		// export folder should contain only folders for language pairs
		// each folder corresponds to a pair of projectors, the folder name should be a string "Lang1Id_Lang2Id"
		TStr ModelFolder = ModelFolder_;
		ModelFolder.ChangeChAll('\\', '/');
		if (ModelFolder[ModelFolder.Len()-1] == '/') {
			ModelFolder = ModelFolder_.GetSubStr(0, ModelFolder.Len()-2);
		}

		TStrV DirNames;
		TStrV Ext(1); Ext[0] = "";
		TFFile::GetFNmV(ModelFolder, Ext, false, DirNames);

		Projectors.Gen(DirNames.Len(), 0);
		Centers.Gen(DirNames.Len(), 0);
		LangIds.Gen(DirNames.Len(), 0);
		SwDispTmMsg("Start loading projectors");
		for (int PairN = 0; PairN < DirNames.Len(); PairN++) {
			printf("%s\n%s\n", DirNames[PairN].CStr(), ModelFolder.CStr());
			//TODO check if DirNames[PairN] is a folder
			TStr PairID = DirNames[PairN].RightOfLast('/');
			printf("%s\n", PairID.CStr());
			TStr Lang1ID;
			TStr Lang2ID;
			PairID.SplitOnCh(Lang1ID, '_', Lang2ID);
			LangIds.Add(TPair<TStr,TStr>(Lang1ID, Lang2ID));
			TFIn CenterFile1(DirNames[PairN] + "/c1.bin");
			TFltV c1(CenterFile1);
			TFIn CenterFile2(DirNames[PairN] + "/c2.bin");
			TFltV c2(CenterFile2);
			Centers.Add(TPair<TFltV,TFltV>(c1, c2));
			TFIn ProjectorFile1(DirNames[PairN] + "/P1.bin");
			TFltVV P1(ProjectorFile1);
			TFIn ProjectorFile2(DirNames[PairN] + "/P2.bin");
			TFltVV P2(ProjectorFile2);
			Projectors.Add(TPair<TFltVV,TFltVV>(P1, P2));

			SwDispTmMsg("Loaded projectors in :" + DirNames[PairN]);
		}
		for (int i = 0; i < DirNames.Len(); i++) {
			printf("%s %s\n", LangIds[i].Val1.CStr(), LangIds[i].Val2.CStr());
		}
	}	
	template <class val, class TSizeTy, bool colmajor>
	inline void TCLHubProjectorAbstract<val, TSizeTy, colmajor>::Load(const TStr& ModelFolder_, const TStr& HubHubProjectorDir) {
		// Important:
		// export folder should contain only folders for language pairs and a folder ConMat
		// each folder corresponds to a pair of projectors, the folder name should be a string "Lang1Id_Lang2Id"
		// ASSUMPTION! : Lang1Id must equal the hub language!
		TStr ModelFolder = ModelFolder_;
		ModelFolder.ChangeChAll('\\', '/');
		if (ModelFolder[ModelFolder.Len()-1] == '/') {
			ModelFolder = ModelFolder_.GetSubStr(0, ModelFolder.Len()-2);
		}

		TStrV DirNames;
		TStrV Ext(1); Ext[0] = "";
		TFFile::GetFNmV(ModelFolder, Ext, false, DirNames);		

		Projectors.Gen(DirNames.Len()-1,0);
		Centers.Gen(DirNames.Len()-1,0);
		InvDoc.Gen(DirNames.Len()-1,0);
		LangIds.Gen(DirNames.Len()-1);
		int PairN = 0;
		for (int DirN = 0; DirN < DirNames.Len(); DirN++) {
			TStr PairID = DirNames[DirN].RightOfLast('/');
			if (PairID.EqI("ConMat")) {
				continue;
			}

			TStr Lang1ID;
			TStr Lang2ID;
			PairID.SplitOnCh(Lang1ID, '_', Lang2ID);

			if (PairID.EqI(HubHubProjectorDir)) {
				HubHubProjectorIdx = PairN;
				HubLangId = Lang1ID;
			}
			LangIds.AddDat(TPair<TStr, TStr>(Lang1ID, Lang2ID), PairN);

			TFIn Center1File(DirNames[DirN] + "/c1.bin");
			TVec<TNum<val>, TSizeTy> c1(Center1File);
			TFIn Center2File(DirNames[DirN] + "/c2.bin");
			TVec<TNum<val>, TSizeTy> c2(Center2File);
			Centers.Add(TPair<TVec<TNum<val>, TSizeTy>, TVec<TNum<val>, TSizeTy>>(c1, c2));
			TFIn idoc1In(DirNames[DirN] + "/invdoc1.bin");
			TVec<TNum<val>, TSizeTy> idoc1(idoc1In);
			TFIn idoc2In(DirNames[DirN] + "/invdoc2.bin");
			TVec<TNum<val>, TSizeTy> idoc2(idoc2In);
			InvDoc.Add(TPair<TFltV, TFltV>(idoc1, idoc2));
			TFIn Projector1File(DirNames[DirN] + "/P1.bin");
			TVVec<TNum<val>, TSizeTy, colmajor> P1;
			if (Lang2ID == "de"){ P1.Load(Projector1File); }
			TFIn Projector2File(DirNames[DirN] + "/P2.bin");
			TVVec<TNum<val>, TSizeTy, colmajor> P2(Projector2File);
			Projectors.Add(TPair<TVVec<TNum<val>, TSizeTy, colmajor>, TVVec<TNum<val>, TSizeTy, colmajor> >(P1, P2));
			PairN++;
		}

		TStr ConPath = ModelFolder + "/ConMat";		
		Ext.Add("bin");
		TStrV FileNames;
		TFFile::GetFNmV(ConPath, Ext, false, FileNames);
		ConMatrices.Gen(FileNames.Len(),0);
		ConIdxH.Gen(FileNames.Len());
		for (int FileN = 0; FileN < FileNames.Len(); FileN++) {
			TPair<TVVec<TNum<val>, TSizeTy, colmajor>, TVVec<TNum<val>, TSizeTy, colmajor>> ConMat;
			TFIn File(FileNames[FileN]);
			ConMat.Load(File);			
			ConMatrices.Add(ConMat);
			printf("%s\n", FileNames[FileN].CStr());

			TStr PairID = FileNames[FileN].RightOfLast('/');
			PairID = PairID.LeftOfLast('.');
			printf("%s\n", PairID.CStr());
			TStr Lang1ID;
			TStr Lang2ID;
			PairID.SplitOnCh(Lang1ID, '_', Lang2ID);
			ConIdxH.AddDat(TPair<TStr,TStr>(Lang1ID, Lang2ID), FileN);
		}	
		
	}

	template <class val, class TSizeTy, bool colmajor>
	inline void TCLHubProjectorAbstract<val, TSizeTy, colmajor>::Load(const TStr& ProjectorPathsDirNm, const TStr& ConMatPathsFNm, const TStr& HubHubProjectorLangPairId, const bool& onlyhub) {
		// use linux paths not windows 
		// remove last / !
		TStr Path;
		if (!TFile::Exists(ProjectorPathsDirNm)){
			printf("Projectors path file %s does not exist", ProjectorPathsDirNm.CStr());
		}
		TFIn ProjectorPathReader(ProjectorPathsDirNm);
		int ProjPairs = 0;
		while (ProjectorPathReader.GetNextLn(Path)) {
			ProjPairs++;
		}		
		ProjectorPathReader.Reset();
        //0->ProjPairs
		Projectors.Gen(ProjPairs, ProjPairs);
		Centers.Gen(ProjPairs, ProjPairs);
		InvDoc.Gen(ProjPairs, ProjPairs);
		LangIds.Gen(ProjPairs);
		int PairN = 0;
		while (ProjectorPathReader.GetNextLn(Path)) {
			printf("Projector path %s", Path.CStr());
			if (Path[Path.Len() -1] == '/') {
				Path = Path.GetSubStr(0, Path.Len() - 2);
			}
			TStr PairID = Path.RightOfLast('/');
			TStr Lang1ID;			
			TStr Lang2ID;
			PairID.SplitOnCh(Lang1ID, '_', Lang2ID);
			if (PairID.EqI(HubHubProjectorLangPairId)) {
				HubHubProjectorIdx = PairN;
				HubLangId = Lang1ID;
			}
			LangIds.AddDat(TPair<TStr,TStr>(Lang1ID, Lang2ID), PairN);
			TPair<TVVec<TNum<val>, TSizeTy, colmajor>, TVVec<TNum<val>, TSizeTy, colmajor>>& ProjectorsPair = Projectors[PairN];
			TPair<TVec<TNum<val>, TSizeTy>, TVec<TNum<val>, TSizeTy>>&   CentersPair = Centers[PairN];
			TPair<TVec<TNum<val>, TSizeTy>, TVec<TNum<val>, TSizeTy>>&   InvDocPair = InvDoc[PairN];

			TFIn Center1File(Path + "/c1.bin");
			//TFltV c1(Center1File);
			CentersPair.Val1.Load(Center1File);
			TFIn Center2File(Path + "/c2.bin");
			//TFltV c2(Center2File);
			CentersPair.Val2.Load(Center2File);
			//Centers.Add(TPair<TFltV,TFltV>(c1, c2));
			///Centers.Add(CentersPair);
			TFIn idoc1In(Path + "/invdoc1.bin");
			//TFltV idoc1(idoc1In);
			InvDocPair.Val1.Load(idoc1In);
			TFIn idoc2In(Path + "/invdoc2.bin");
			//TFltV idoc2(idoc2In);
			InvDocPair.Val2.Load(idoc2In);
			//InvDoc.Add(TPair<TFltV, TFltV>(idoc1, idoc2));
			//InvDoc.Add(InvDocPair);
			TStr Friend = "de";
			TFIn Projector1File(Path + "/P1.bin");
			//TFltVV P1(Projector1File);
			//Andrej: we need only projector for a friend
			if (Lang2ID == Friend || onlyhub == false ){
				ProjectorsPair.Val1.Load(Projector1File);
			}
            TFIn Projector2File(Path + "/P2.bin");
			//TFltVV P2(Projector2File);
			ProjectorsPair.Val2.Load(Projector2File);
			//Projectors.Add(TPair<TFltVV,TFltVV>(P1, P2));
			//Projectors.Add(ProjectorsPair);

			PairN++;
			//printf("loaded hub projector %s, dims %d %d %d %d %d %d %d %d\n", Path.CStr(), c1.Len(), c2.Len(), idoc1.Len(), idoc2.Len(), P1.GetRows(), P1.GetCols(), P2.GetRows(), P2.GetCols());
		}
		printf("Hubs OK\n");
		TInt ConMatrixCount = 0;
		TFIn ConMatricesPathReader(ConMatPathsFNm);
		while (ConMatricesPathReader.GetNextLn(Path)) {
			ConMatrixCount++;
		}
		ConMatricesPathReader.Reset();
        //0->ConMatrixCount
		ConMatrices.Gen(ConMatrixCount, ConMatrixCount);
		ConIdxH.Gen(ConMatrixCount);
		int FileN = 0;
		printf("Starting and reading conmat %s\n", ConMatPathsFNm.CStr());
		while (ConMatricesPathReader.GetNextLn(Path)) {		
			printf("Given path: %s\n", Path.CStr());
			TPair< TVVec<TNum<val>, TSizeTy, colmajor>, TVVec<TNum<val>, TSizeTy, colmajor> >& ConMat = ConMatrices[FileN];
			//Check if File exists and then proceed with loading
			TStr PairID = Path.RightOfLast('/');
			TStr AlternatePath = Path.LeftOfLast('/');
			TStr Ending = PairID.RightOfLast('.');
			PairID = PairID.LeftOfLast('.');
			printf("%s\n", PairID.CStr());
			TStr Lang1ID;
			TStr Lang2ID;
			PairID.SplitOnCh(Lang1ID, '_', Lang2ID);
			AlternatePath = AlternatePath + "/" + Lang2ID + "_" + Lang1ID + "." + Ending;
			if (!TFile::Exists(Path)){
				if (TFile::Exists(AlternatePath)){
					printf("Using alternate path: %s\n", Path.CStr());
					printf("Swapping language ids!\n");
					TStr Temp = Lang2ID;
					Lang2ID = Lang1ID;
					Lang1ID = Temp;
					Path = AlternatePath;
				}
			}
			//Andrej: only load connection matrices with friend
			TStr Friend = "de";
			if (!(Lang1ID == Friend || Lang2ID == Friend) && onlyhub){
				continue;
			}
			TFIn File(Path);
			ConMat.Load(File);
            //For components of ConMatrices following should be valid
            //If we want that they represent the scalar product
            // Q1 and Q2 have orthonormal (rows, columns) and Q2 = Q1'
            // Therefore this can be optimised for common en-de approach
            // en-sl to en-de needs only sl-de matrix! 			
			//ConMatrices.Add(ConMat);
			printf("%s %g %g %d %d %g %g %d %d\n", Path.CStr(), ConMat.Val1(0, 0).Val, ConMat.Val1(0, 1).Val, ConMat.Val1.GetRows(), ConMat.Val1.GetCols(), ConMat.Val2(0, 0).Val, ConMat.Val2(0, 1).Val, ConMat.Val2.GetRows(), ConMat.Val2.GetCols());
			ConIdxH.AddDat(TPair<TStr,TStr>(Lang1ID, Lang2ID), FileN);

			printf("loaded conmat %s\n", Path.CStr());

			FileN++;
			//char c;
			//std::cin >> c;
		}
	}

	// Maps language identifiers (DocLangId, Lang2Id) to Projector, Center, ConMat
	// ASSUMPTION: All Projector data has been loaded and will stay there (references are passed)
	//[Andrej] This could be improved, similarity function should be symmetric, computing sim between en - de or de - en docs should always return the same result!!! 
	// Allready partially implemented in GetProxyMatrix 
	template <class val, class TSizeTy, bool colmajor>
	inline bool TCLHubProjectorAbstract<val, TSizeTy, colmajor>::SelectMatrices(const TStr& DocLangId, const TStr& Lang2Id, const TVVec<TNum<val>, TSizeTy, colmajor>*& ProjMat, const TVec<TNum<val>, TSizeTy>*& Center, const TVVec<TNum<val>, TSizeTy, colmajor>*& ConMat, const TVec<TNum<val>, TSizeTy>*& InvDocV,  const TBool& DmozSpecial) const
	{
	bool OK = true;
		// Give reference to Projector and Center and ConMat
		//printf("\n\n%s %d\n\n", HubLangId.CStr(), HubHubProjectorIdx);
		//printf ("%d ffekjffoief\n", HubHubProjectorIdx);
		bool DocHub = DocLangId.EqI(HubLangId);
		bool OtherHub = Lang2Id.EqI(HubLangId);
		int ProjIdx;
		
		// Val1
		// hub - hub
		if (DocHub && OtherHub) {
			//printf("DOCHUB OTHERHUB!\n");
			ProjMat = &Projectors[HubHubProjectorIdx].Val1;
			Center = &Centers[HubHubProjectorIdx].Val1;
			InvDocV = &InvDoc[HubHubProjectorIdx].Val1;
		}
		// hub - other
		if (DocHub && !OtherHub) {
			//printf("DocHUb other not hub\n");
			int KeyId;
			//if (LangIds.IsKey(TStrPr(DocLangId, Lang2Id), KeyId)) {
			if (LangIds.IsKey(TStrPr(HubLangId, Lang2Id), KeyId)) {
				ProjIdx = LangIds[KeyId];
				ProjMat = &Projectors[ProjIdx].Val1;
				//printf("%d\n", ProjMat);
				Center = &Centers[ProjIdx].Val1;
				InvDocV = &InvDoc[ProjIdx].Val1;
			} else {
				printf("hub - other false!\n");
				printf("DocLangId: %s\n", DocLangId.CStr());
				OK = false;
				/*if (!DmozSpecial){
					//printf("ok false in dochub && !otherdoc uh. ARE YOU MISSING A PATH IN hubprojectors.txt IN CONFIG FOLDER?\n");
				}
				else{
					//printf("I am assuming identity matrix by default, i hope i am right?\n");
				}*/
			}
		}
		// Val2
		// other - hub
		if (!DocHub && OtherHub) {
			int KeyId;
			if (LangIds.IsKey(TStrPr(HubLangId, DocLangId), KeyId)) {
				ProjIdx = LangIds[KeyId];
				ProjMat = &Projectors[ProjIdx].Val2;
				Center = &Centers[ProjIdx].Val2;
				InvDocV = &InvDoc[ProjIdx].Val2;
			}
			else {
				printf("other - hub false!\n");
				printf("DocLangId: %s\n", DocLangId.CStr());
				OK = false;
				/*if (!DmozSpecial){
					//printf("ok false in !dochub && otherdocuh. ARE YOU MISSING A PATH IN hubprojectors.txt IN CONFIG FOLDER?\n");
				}
				else{
					//printf("I am assuming identity matrix by default, i hope i am right?\n");
				}*/
				
			}
		}
		// other - other
		if (!DocHub && !OtherHub) {
			//printf("!DOCHUB !OTHERHUB!\n");
			int KeyId;
			/*Andrej Debug Heavy: */
			if (LangIds.IsKey(TStrPr(HubLangId, DocLangId), KeyId)) {
				ProjIdx = LangIds[KeyId];
				ProjMat = &Projectors[ProjIdx].Val2;
				Center = &Centers[ProjIdx].Val2;
				InvDocV = &InvDoc[ProjIdx].Val2;
			} else {
				TVec<TKeyDat<TStrPr, TInt>, int> test;
				LangIds.GetKeyDatKdV(test);
				printf("DocLangId %s; Lang2Id %s\n", DocLangId.CStr(), Lang2Id.CStr());
				for (int i = 0; i < test.Len(); i++){
					printf("Key: %s; Dat: %s", test[i].Key.GetStr().CStr(), test[i].Dat.GetStr().CStr());
				}
				OK = false;
				printf("ok false in !dochub && !otherdocuh. ARE YOU MISSING A PATH IN hubprojectors.txt IN CONFIG FOLDER?\n");
			}
		}
		// Find DocLangId, Lang2Id in ConMat
		int KeyId;
		if (ConIdxH.IsKey(TStrPr(DocLangId, Lang2Id), KeyId)) {
			ConMat = &ConMatrices[ConIdxH[KeyId]].Val1;
		// Else, find Lang2Id, DocLangId in Conmat
		} else if (ConIdxH.IsKey(TStrPr(Lang2Id, DocLangId), KeyId)) {
			ConMat = &ConMatrices[ConIdxH[KeyId]].Val2;
		} else {
			OK = false;
			printf("ok false in conmat, ARE YOU MISSING A PATH IN conmat.txt IN CONFIG FOLDER? %d %d\n", DocHub, OtherHub);
			/*Andrej Debug Heavy: */
			TVec<TKeyDat<TStrPr, TInt>, int> test;
			LangIds.GetKeyDatKdV(test);
			printf("DocLangId %s; Lang2Id %s\n", DocLangId.CStr(), Lang2Id.CStr());
			for (int i = 0; i < test.Len(); i++){
				printf("Key: %s; Dat: %s", test[i].Key.GetStr().CStr(), test[i].Dat.GetStr().CStr());
			}
			OK = false;
			printf("ok false in !dochub && !otherdocuh. ARE YOU MISSING A PATH IN hubprojectors.txt IN CONFIG FOLDER?\n");
		}
		return OK;
	}
    //Andrej Improve logic
    template <class val, class TSizeTy, bool colmajor>
	inline bool TCLHubProjectorAbstract<val, TSizeTy, colmajor>::SelectMatrices(const TStr& DocLangId, const TPair<TStr, TStr>& TargetSpace, const TVVec<TNum<val>, TSizeTy, colmajor>*& ProjMat, const TVec<TNum<val>, TSizeTy>*& Center, const TVVec<TNum<val>, TSizeTy, colmajor>*& ConMat, const TVec<TNum<val>, TSizeTy>*& InvDocV, TBool& transpose) const
	{
		bool OK = true;
		// Give reference to Projector and Center and ConMat
		//printf("\n\n%s %d\n\n", HubLangId.CStr(), HubHubProjectorIdx);
		//printf ("%d ffekjffoief\n", HubHubProjectorIdx);
		TStr Target1 = TargetSpace.Val1;
		TStr Target2 = TargetSpace.Val2;
		if (Target1 != HubLangId && Target2 != HubLangId){
			OK = false;
			ErrNotify("HubApproach: at least one component of common space " + Target1 + "_" + Target2 + " should be " + " Hub: " + HubLangId);
			return OK;
		}
		bool DocHub = DocLangId.EqI(HubLangId);
		bool OtherHubFirst = Target1.EqI(HubLangId);
		if (!OtherHubFirst){
			Swap(Target1, Target2);
			OtherHubFirst = true;
		}
		int ProjIdx;

		if (DocHub) {
			//printf("DocHUb other not hub\n");
			int KeyId;
			//if (LangIds.IsKey(TStrPr(DocLangId, Lang2Id), KeyId)) {
			if (LangIds.IsKey(TStrPr(HubLangId, Target2), KeyId)) {
				ProjIdx = LangIds[KeyId];
				ProjMat = &Projectors[ProjIdx].Val1;
				//printf("%d\n", ProjMat);
				Center = &Centers[ProjIdx].Val1;
				InvDocV = &InvDoc[ProjIdx].Val1;
			}
			else {
				OK = false;
				/*if (!DmozSpecial){
					//printf("ok false in dochub && !otherdoc uh. ARE YOU MISSING A PATH IN hubprojectors.txt IN CONFIG FOLDER?\n");
				}
				else{
					//printf("I am assuming identity matrix by default, i hope i am right?\n");
				}*/
			}
		}
		// Val2
		// other - hub
		if (!DocHub) {
			int KeyId;
			if (LangIds.IsKey(TStrPr(HubLangId, DocLangId), KeyId)) {
				ProjIdx = LangIds[KeyId];
				ProjMat = &Projectors[ProjIdx].Val2;
				Center = &Centers[ProjIdx].Val2;
				InvDocV = &InvDoc[ProjIdx].Val2;
			}
			else {
				OK = false;
				/*if (!DmozSpecial){
					//printf("ok false in !dochub && otherdocuh. ARE YOU MISSING A PATH IN hubprojectors.txt IN CONFIG FOLDER?\n");
				}
				else{
					//printf("I am assuming identity matrix by default, i hope i am right?\n");
				}*/

			}
		}
		// Find DocLangId, Lang2Id in ConMat
		int KeyId;
		if (ConIdxH.IsKey(TStrPr(DocLangId, Target2), KeyId)) {
			ConMat = &ConMatrices[ConIdxH[KeyId]].Val1;
			// Else, find Target2, DocLangId in Conmat
		}
		else if (ConIdxH.IsKey(TStrPr(Target2, DocLangId), KeyId)) {
			ConMat = &ConMatrices[ConIdxH[KeyId]].Val1;
		}
		else {
			OK = false;
			printf("ok false in conmat, ARE YOU MISSING A PATH IN conmat.txt IN CONFIG FOLDER? %d %d\n", DocHub, OtherHubFirst);
		}
		return OK;
	}

	//Andrej check
	template <class val, class TSizeTy, bool colmajor>
	inline bool TCLHubProjectorAbstract<val, TSizeTy, colmajor>::GetCenters(const TStr& DocLangId_, const TStr& Lang2Id_, const TVec<TNum<val>, TSizeTy>*& Center1, const TVec<TNum<val>, TSizeTy>*& Center2, const TStr& Friend) const{
		TStr DocLangId = DocLangId_;
		TStr Lang2Id = Lang2Id_;
		if (Lang2Id == HubLangId)
			Lang2Id = Friend;
		int KeyId;
		int ProjIdx;
		if (LangIds.IsKey(TStrPr(HubLangId, Lang2Id), KeyId)) {
#ifdef DEBUG_ENABLE
			printf("Centers %s\n", Lang2Id.CStr());
#endif
			ProjIdx = LangIds[KeyId];
			//printf("%d\n", ProjMat);
			Center1 = &Centers[ProjIdx].Val1;
			Center2 = &Centers[ProjIdx].Val2;
		}
		else{
			return false;
		}
		return true;
	}
    
	template <class val, class TSizeTy, bool colmajor>
	inline bool TCLHubProjectorAbstract<val, TSizeTy, colmajor>::GetCenter(const TStr& DocLangId, const TPair<TStr, TStr>& TargetSpace, const TVec<TNum<val>, TSizeTy>*& Center) const{
		int KeyId;
		int ProjIdx;
		EAssert(TargetSpace.Val1 == HubLangId);
		if (LangIds.IsKey(TargetSpace, KeyId)) {
				ProjIdx = LangIds[KeyId];
				if (DocLangId == HubLangId){
					if (DocLangId == TargetSpace.Val1){
						Center = &Centers[ProjIdx].Val1;
					}
					else{
						if (DocLangId == TargetSpace.Val2){
							Center = &Centers[ProjIdx].Val2;
                        }
					}

				}

		}
		
 
		return true;
	}

	template <class val, class TSizeTy, bool colmajor>
	inline bool TCLHubProjectorAbstract<val, TSizeTy, colmajor>::GetProxyMatrix(const TStr& DocLangId_, const TStr& Lang2Id_, const TFltVV*& ConMat, bool &transpose_flag, const TStr& Friend) const{
		bool OK = true;
		transpose_flag = false;
		TStr DocLangId = DocLangId_;
		TStr Lang2Id   = Lang2Id_;
		if (DocLangId_.EqI(HubLangId)){
			//Going from en-es, en projected to en-Friend
			//Connect with Friend-es
			DocLangId = Friend;
		}
		if (Lang2Id_.EqI(HubLangId)){
			Lang2Id = Friend;
		}
		int KeyId;
		if (ConIdxH.IsKey(TStrPr(DocLangId, Lang2Id), KeyId)) {
			ConMat = &ConMatrices[ConIdxH[KeyId]].Val1;
			transpose_flag = true;
			// Else, find Lang2Id, DocLangId in Conmat
		}
		else if (ConIdxH.IsKey(TStrPr(Lang2Id, DocLangId), KeyId)) {
			ConMat = &ConMatrices[ConIdxH[KeyId]].Val1;
			transpose_flag = false;
		}
		else {
			OK = false;
			//printf("ok false in conmat, ARE YOU MISSING A PATH IN conmat.txt IN CONFIG FOLDER? %d %d\n", DocHub, OtherHub);
		}
		return OK;
	}


	template <class val, class TSizeTy, bool colmajor>
	inline void TCLHubProjectorAbstract<val, TSizeTy, colmajor>::DoTfidf(TSparseMatrix<val, TSizeTy>& DocMatrix, const TStr& DocLangId, const TStr& Lang2Id){
		const TFltVV* ProjMat = NULL;
		const TFltV* Center = NULL;
		const TFltVV* ConMat = NULL;
		const TFltV* InvDocV = NULL;
		bool OK = TCLHubProjector::SelectMatrices(DocLangId, Lang2Id, ProjMat, Center, ConMat, InvDocV);
		if (!OK) printf("Select failed\n");
		TTmStopWatch time;
		for (int ElN = 0; ElN < DocMatrix.Val1.Len(); ElN++) {
			DocMatrix.Val3[ElN] = DocMatrix.Val3[ElN] * (*InvDocV)[DocMatrix.Val1[ElN]];
		}
	}

	template <class val, class TSizeTy, bool colmajor>
	inline void TCLHubProjectorAbstract<val, TSizeTy, colmajor>::DoTfidf(TSparseVec<val, TSizeTy> & Doc, const TStr& DocLangId, const TStr& Lang2Id){
		const TFltVV* ProjMat = NULL;
		const TFltV* Center = NULL;
		const TFltVV* ConMat = NULL;
		const TFltV* InvDocV = NULL;
		bool OK = TCLHubProjector::SelectMatrices(DocLangId, Lang2Id, ProjMat, Center, ConMat, InvDocV);
		if (!OK) printf("Select failed\n");
		TTmStopWatch time;
		time.Start();
		for (int ElN = 0; ElN < Doc.Val1.Len(); ElN++) {
			Doc.Val2[ElN] = Doc.Val2[ElN] * (*InvDocV)[Doc.Val1[ElN]];
		}
		time.Stop("Inverse indexing costs: ");
	}

	template <class val, class TSizeTy, bool colmajor>
	inline TCLCoreAbstract<val, TSizeTy, colmajor>::TCLCoreAbstract(TCLProjectorAbstract<val, TSizeTy, colmajor>* Projectors_, const TStrV& TokenzierLangIdV, const TStrV& TokenizerPaths) {
		printf("Tokenizers start\n");
		Projectors = Projectors_;
		Tokenizers.Gen(TokenizerPaths.Len());
		for (int TokenizerN = 0; TokenizerN < TokenizerPaths.Len(); TokenizerN++) {
			TUnicodeVSM::PGlibUBow Tokenizer = TUnicodeVSM::PGlibUBow::New();
			printf("Tokenizers file %s\n", TokenizerPaths[TokenizerN].CStr());
			TFIn TokenizerFile(TokenizerPaths[TokenizerN]);

			Tokenizer->LoadBin(TokenizerFile);
			Tokenizers.AddDat(TokenzierLangIdV[TokenizerN], Tokenizer);			
		}
		printf("Tokenizers done\n");
	}

	template <class val, class TSizeTy, bool colmajor>
	inline TCLCoreAbstract<val, TSizeTy, colmajor>::TCLCoreAbstract(TCLProjectorAbstract<val, TSizeTy, colmajor>* Projectors_, const TStr& TokenizerPathsFNm) {
		Projectors = Projectors_;
		TFIn TokenizerReader(TokenizerPathsFNm);
		TStrV TokenizerPathV;
		TStr Path;
		while (TokenizerReader.GetNextLn(Path)) {
			// extract 
			TStr LangId = Path.RightOfLast('/');
			LangId = LangId.LeftOf('.');
			TUnicodeVSM::PGlibUBow Tokenizer = TUnicodeVSM::PGlibUBow::New();
			TFIn TokenizerFile(Path);
			Tokenizer->LoadBin(TokenizerFile);
			Tokenizers.AddDat(LangId, Tokenizer);
			printf("langid %s\n", LangId.CStr());
			printf("loaded tokenizer %s\n", Path.CStr());
		}		
	}
	template <class val, class TSizeTy, bool colmajor>
	inline void TCLCoreAbstract<val, TSizeTy, colmajor>::TextToVector(TUStr& Text, const TStr& LangId, TSparseVec<TSizeTy, val>& SparseVec) {
		if (Tokenizers.IsKey(LangId)){
			Tokenizers.GetDat(LangId)->TextToVec(Text, SparseVec);
		}
	}
	template <class val, class TSizeTy, bool colmajor>
	inline void TCLCoreAbstract<val, TSizeTy, colmajor>::TextToVector(TUStrV& Docs, const TStr& LangId, TSparseMatrix<TSizeTy, val>& DocMatrix) {
		if (Tokenizers.IsKey(LangId)){
			Tokenizers.GetDat(LangId)->TextToVec(Docs, DocMatrix);
		}
	}

	
	template <class val, class TSizeTy, bool colmajor>
	inline void TCLCoreAbstract<val, TSizeTy, colmajor>::Project(TUStr& Text, const TStr& DocLangId, const TStrPr& TargetSpace, TVec<TNum<val>, TSizeTy>& Projected, const TBool& DmozSpecial) {
		TPair<TIntV, TFltV> SparseVec;
		TextToVector(Text, DocLangId, SparseVec);
		Project(SparseVec, DocLangId, TargetSpace, Projected, false);
	}

	template <class val, class TSizeTy, bool colmajor>
	inline void TCLCoreAbstract<val, TSizeTy, colmajor>::Project(const TSparseVec<TSizeTy, val>& Doc, const TStr& DocLangId, const TStrPr& TargetSpace, TVec<TNum<val>, TSizeTy>& Projected, const TBool& DmozSpecial) {
		Projectors->Project(Doc, DocLangId, TargetSpace, Projected, DmozSpecial);
	}
	
	//Mark
	template <class val, class TSizeTy, bool colmajor>
	inline void TCLCoreAbstract<val, TSizeTy, colmajor>::Project(TUStrV& Docs, const TStr& DocLangId, const TStrPr& TargetSpace, TVVec<TNum<val>, TSizeTy, colmajor>& ProjectedMat, const TBool& DmozSpecial) {
		TTriple<TIntV, TIntV, TFltV> DocMatrix;
		TextToVector(Docs, DocLangId, DocMatrix);
		//[Andrej] DmozSpecial and TFIdf Troubles
		Projectors->Project(DocMatrix, DocLangId, TStrPr("en","de"), ProjectedMat, true, DmozSpecial);
	}
	template <class val, class TSizeTy, bool colmajor>
	inline void TCLCoreAbstract<val, TSizeTy, colmajor>::Project(const TSparseMatrix<TSizeTy, val>& DocMatrix, const TStr& DocLangId, const TStrPr& TargetSpace, TVVec<TNum<val>, TSizeTy, colmajor>& Projected, const TBool& DmozSpecial) {
		Projectors->Project(DocMatrix, DocLangId, TStrPr("en", "de"), Projected, true, DmozSpecial);
	}

	template <class val, class TSizeTy, bool colmajor>
	inline double TCLCoreAbstract<val, TSizeTy, colmajor>::GetSimilarity(TUStr& Text1, TUStr& Text2, const TStr& Lang1Id, const TStr& Lang2Id) {
#ifdef DEBUG_ENABLE
		printf("\nDoc1: %s\n Doc2: %s\n", Text1.GetStr().CStr(), Text2.GetStr().CStr());
#endif
		TPair<TIntV, TFltV> Doc1;
		TPair<TIntV, TFltV> Doc2;		
		TextToVector(Text1, Lang1Id, Doc1);
		TextToVector(Text2, Lang2Id, Doc2);
		if (Doc1.Val1.Len() == 0 || Doc2.Val1.Len() == 0){
			return 0;
		}
		return GetSimilarity(Doc1, Doc2, Lang1Id, Lang2Id);
	}

	template <class val, class TSizeTy, bool colmajor>
	inline double TCLCoreAbstract<val, TSizeTy, colmajor>::GetSimilarity(const TSparseVec<TSizeTy, val>& Doc1, const TSparseVec<TSizeTy, val>& Doc2, const TStr& Lang1Id, const TStr& Lang2Id){
		if (Doc1.Val1.Len() == 0 || Doc2.Val2.Len() == 0){
			return 0;
		}
		TFltV PDoc1; TFltV PDoc2;
		Project(Doc1, Lang1Id, TStrPr("en", "de"), PDoc1, false);
		Project(Doc2, Lang2Id, TStrPr("en", "de"), PDoc2, false);
		//This could counter the centering missmatch but it does not matter in the practice
		/*for (int i = 0; i < PDoc2.Len(); i++){
			PDoc2[i] = PDoc2[i] + (*Center1en)[i] - (*Center2en)[i];
		}*/
			//->SelectMatrices(const TStr& DocLangId, const TStr& Lang2Id, const TFltVV*& ProjMat, const TFltV*& Center, const TFltVV*& ConMat, const TFltV*& InvDocV, const TBool& DmozSpecial)
		TLinAlg::Normalize(PDoc1);
		TLinAlg::Normalize(PDoc2);
		return TLinAlg::DotProduct(PDoc1, PDoc2);		
	}

	//Take Care of Memory management inside
	template <class val, class TSizeTy, bool colmajor>
	inline void TCLCoreAbstract<val, TSizeTy, colmajor>::GetSimilarity(const TSparseMatrix<TSizeTy, val>& DocMtx1, const TSparseMatrix<TSizeTy, val>& DocMtx2, const TStr& Lang1Id, const TStr& Lang2Id, TVVec<TNum<val>, TSizeTy, colmajor>& SimMtx){
		// PDocMtx1 should be transposed to obtain better performance
		TFltVV PDocMtx1;  Project(DocMtx1, Lang1Id, Lang2Id, PDocMtx1);
		//printf("(%d, %d)\n", PDocMtx1.GetXDim(), PDocMtx1.GetYDim());
		TFltVV PDocMtx2;  Project(DocMtx2, Lang2Id, Lang1Id, PDocMtx2);
		//printf("(%d, %d)\n", PDocMtx2.GetXDim(), PDocMtx2.GetYDim());
		TLinAlg::NormalizeData(PDocMtx1);
		TLinAlg::NormalizeData(PDocMtx2);
		//printf("(%d, %d)\n", SimMtx.GetXDim(), SimMtx.GetYDim());
		return TLinAlg::MultiplyT(PDocMtx1, PDocMtx2, SimMtx);
	}


	// Returns top k words in lang1 and top k words in lang2 that contributed to the similarity
	template <class val, class TSizeTy, bool colmajor>
	inline double TCLCoreAbstract<val, TSizeTy, colmajor>::ExplainSimilarity(const TSparseVec<TSizeTy, val>& Doc1, const TSparseVec<TSizeTy, val>& Doc2, const TStr& Lang1Id, const TStr& Lang2Id, TVec<TPair<TNum<val>, TNum<TSizeTy>>, TSizeTy>& Lang1Words, TVec<TPair<TNum<val>, TNum<TSizeTy>>, TSizeTy>& Lang2Words) {
		int n1 = Doc1.Val1.Len();
		int n2 = Doc2.Val1.Len();
		// Transform the Doc into a matrix (each word gets mapped to a weighted indicator vector, a column in the matrix)
		TTriple<TIntV, TIntV, TFltV> Doc1Mat;

		Doc1Mat.Val1.Gen(n1, 0);
		Doc1Mat.Val2.Gen(n1, 0);
		Doc1Mat.Val3.Gen(n1, 0);
		for (int ElN = 0; ElN < n1; ElN++) {
			Doc1Mat.Val1.Add(Doc1.Val1[ElN]);
			Doc1Mat.Val2.Add(ElN);
			Doc1Mat.Val3.Add(Doc1.Val2[ElN]);
		}
		TTriple<TIntV, TIntV, TFltV> Doc2Mat;
		Doc2Mat.Val1.Gen(n2, 0);
		Doc2Mat.Val2.Gen(n2, 0);
		Doc2Mat.Val3.Gen(n2, 0);
		
		for (int ElN = 0; ElN < n2; ElN++) {
			Doc2Mat.Val1.Add(Doc2.Val1[ElN]);
			Doc2Mat.Val2.Add(ElN);
			Doc2Mat.Val3.Add(Doc2.Val2[ElN]);
		}
		TFltVV PDoc1Mat;
		TFltV PDoc1; 
		TFltVV PDoc2Mat;
		TFltV PDoc2;

		Project(Doc1Mat, Lang1Id, TStrPr("en", "de"), PDoc1Mat, false);
		//TLinAlg::NormalizeData(PDoc1Mat);
		Project(Doc1, Lang1Id, TStrPr("en", "de"), PDoc1, false);

		Project(Doc2Mat, Lang2Id, TStrPr("en", "de"), PDoc2Mat, false);
		//TLinAlg::NormalizeData(PDoc2Mat);
		Project(Doc2, Lang2Id, TStrPr("en", "de"), PDoc2, false);
		
		TFltV Sim1(n1, n1);
		TFltV Sim2(n2, n2);

#ifdef COLMAJOR_DATA
		TLinAlg::MultiplyT(PDoc1Mat, PDoc2, Sim1);
		TLinAlg::MultiplyT(PDoc2Mat, PDoc1, Sim2);
#else		
		TLinAlg::Multiply(PDoc1Mat, PDoc2, Sim1);
		TLinAlg::Multiply(PDoc2Mat, PDoc1, Sim2);
#endif
		Lang1Words.Gen(n1, 0);
		for (int ElN = 0; ElN < n1; ElN++) {			
			Lang1Words.Add(TPair<TFlt, TInt>(Sim1[ElN], Doc1.Val1[ElN]));
		}	
		
		Lang2Words.Gen(n2, 0);
		for (int ElN = 0; ElN < n2; ElN++) {
			Lang2Words.Add(TPair<TFlt, TInt>(Sim2[ElN], Doc2.Val1[ElN]));
		}
		
		Lang1Words.Sort(false);
		Lang2Words.Sort(false);
		/*
		double sim1 = TLinAlg::SumVec(Sim1);
		double sim2 = TLinAlg::SumVec(Sim2);
		*/
		TLinAlg::Normalize(PDoc1);
		TLinAlg::Normalize(PDoc2);
		return TLinAlg::DotProduct(PDoc1,PDoc2);
	}

	
	// Returns top k words as TStr in lang1 and top k words in lang2 that contributed to the similarity
	template <class val, class TSizeTy, bool colmajor>
	inline double TCLCoreAbstract<val, TSizeTy, colmajor>::ExplainSimilarity(TUStr& Text1, TUStr& Text2, const TStr& Lang1Id, const TStr& Lang2Id, TUStrV& Lang1TopWords, TUStrV& Lang2TopWords, const int& k = 10) {
		TPair<TIntV, TFltV> Doc1;
		TPair<TIntV, TFltV> Doc2;
		TextToVector(Text1, Lang1Id, Doc1);
		TextToVector(Text2, Lang2Id, Doc2);

		if (Doc1.Val1.Len() == 0 || Doc2.Val1.Len() == 0){
			return 0;
		}
		TVec<TPair<TFlt,TInt> > Lang1Words;
		TVec<TPair<TFlt,TInt> > Lang2Words;
		double sim =  ExplainSimilarity(Doc1, Doc2, Lang1Id, Lang2Id, Lang1Words, Lang2Words);	
		Lang1TopWords.Gen(0);		
		for (int WordN = 0; WordN < TMath::Mn(k, Lang1Words.Len()); WordN++) {
			Lang1TopWords.Add(GetWordByKeyIdLangId(Lang1Id, Lang1Words[WordN].Val2));
		}
		Lang2TopWords.Gen(0);
		for (int WordN = 0; WordN < TMath::Mn(k, Lang2Words.Len()); WordN++) {
			Lang2TopWords.Add(GetWordByKeyIdLangId(Lang2Id, Lang2Words[WordN].Val2));
		}

		return sim;

	}

	template <class val, class TSizeTy, bool colmajor>
	inline void TCLCNodeAbstract<val, TSizeTy, colmajor>::Save(TSOut& SOut) const {
		DocIdxV.Save(SOut);
		CatStr.Save(SOut);
		CatPathStr.Save(SOut);
		StrToChildIdH.Save(SOut);
	}

	template <class val, class TSizeTy, bool colmajor>
	inline void TCLCNodeAbstract<val, TSizeTy, colmajor>::Disp() { printf("cat:%s, catpath:%s, docs:%d, children:%d\n", CatStr.CStr(), CatPathStr.CStr(), DocIdxV.Len(), StrToChildIdH.Len()); }

	//Work in progress, paths are hardcoded here
	template <class val = double, class TSizeTy = int, bool colmajor = false>
	inline void ProcessDmozSaveBinarySparse() {
		TStr ConfigFolder = "C:/DD/Twitter/config.jan/Transpose/tokenizers/";
		TStr DmozDataPath = "C:/DD/Twitter/config.jan/Transpose/dmozdata/";
		printf("start loading\n");
		TCrossLingual::TCLHubProjectorAbstract<val, TSizeTy, colmajor> HubProj;
		//HubProj.Load(ConfigFolder + "/hubprojectors.txt", ConfigFolder + "/conmat.txt", "en_es");
		HubProj.LoadHub("C:/DD/Twitter/config.jan/Transpose/hubprojectors/", "C:/DD/Twitter/config.jan/Transpose/simple_conmat/", TStrPr("en", "de"));
		TCrossLingual::TCLCoreAbstract<val, TSizeTy, colmajor> Core(&HubProj, TStrV::GetV( "en", "de" ), TStrV::GetV("C:/DD/Twitter/config.jan/Transpose/tokenizers/en.ubow", "C:/DD/Twitter/config.jan/Transpose/tokenizers/de.ubow"));
		SwDispTmMsg("Core loaded");

		TFIn DmozLnDocReader(DmozDataPath + "/DMozLnDocs.Txt");
		TSizeTy DocN = 0;
		TStr DocStr = "";
		TSparseMatrix<TSizeTy, val> Matrix;

		TSparseVec<TSizeTy, val> SpVec;
		SwDispTmMsg("start processing dmoz");
		TUStrV Docs;
		while (DmozLnDocReader.GetNextLn(DocStr)) {
			if (DocN % 10000 == 0) printf("%d\r", DocN);
			DocStr = DocStr.RightOf('\t');
			TUStr UDocStr(DocStr);
			Docs.Add(UDocStr);
			++DocN;
			/*Core.TextToVector(UDocStr, "en", SpVec);
			if (DocN == 450005) {
				printf("THE DOC: %s\n", DocStr.CStr());
				for (int ElN = 0; ElN < SpVec.Val1.Len(); ElN++) {
					printf("%d %f\n", SpVec.Val1[ElN], SpVec.Val2[ElN]);
				}
			}
			for (int ElN = 0; ElN < SpVec.Val1.Len(); ElN++) {
				Matrix.Val1.Add(SpVec.Val1[ElN]);
				Matrix.Val2.Add(DocN);
				Matrix.Val3.Add(SpVec.Val2[ElN]);
			}
			*/
		}
		Core.TextToVector(Docs, "en", Matrix);

		SwDispTmMsg("done processing dmoz");
		TFOut DmozMatrixWritter(DmozDataPath + "/DmozMatrix.bin");
		Matrix.Save(DmozMatrixWritter);
		SwDispTmMsg("done saving dmoz matrix");
	}

	template <class val, class TSizeTy, bool colmajor>
	inline void TCLClassifierAbstract<val, TSizeTy, colmajor>::LoadData() {
		printf("%s\n", (DmozPath + "/DmozCategories.Bin").CStr());
		TFIn CatFile(DmozPath + "/DmozCategories.Bin");
		Cat.Load(CatFile);
		if (CutDepth > -1) {
			for (int DocN = 0; DocN < Cat.Len(); DocN++) {
				TStrV CatParts;
				Cat[DocN].SplitOnAllCh('/', CatParts, true);
				TStr NewCat = "";
				//Andrej
				//int stopN = TMath::Mn(CutDepth, CatParts.Len());
				int stopN = MIN(CutDepth, CatParts.Len());
				for (int k = 0; k < stopN; k++) {
					NewCat += CatParts[k] + "/";
				}
				Cat[DocN] = NewCat;
			}
		}
		if (CutDepth == -2) {
			for (int DocN = 0; DocN < Cat.Len(); DocN++) {
				Cat[DocN] = "Top/" + TInt::GetStr(DocN) + "/";
			}

		}
		printf("%s\n", (DmozPath + "/DmozVectorDocsNew.Bin").CStr());
		TFIn VectorsNewFile(DmozPath + "/DmozVectorDocsNew.Bin");
		CatDocs.Load(VectorsNewFile);	
		SwDispTmMsg("Done loading dmoz");
	}

	template <class val, class TSizeTy, bool colmajor>
	inline void TCLClassifierAbstract<val, TSizeTy, colmajor>::LoadModel(){
		TFIn CentroidReader(ModelPath + "/Centroids.bin");
		Centroids.Load(CentroidReader);
		SwDispTmMsg("Number of centroids = " + TInt::GetStr(Centroids.GetRows()) + " " + TInt::GetStr(Centroids.GetCols()));
		TFIn TaxonomyReader(ModelPath + "/Taxonomy.bin");
		Taxonomy.Load(TaxonomyReader);
	}

	template <class val, class TSizeTy, bool colmajor>
	inline void TCLClassifierAbstract<val, TSizeTy, colmajor>::ProjectDmoz() {
		SwDispTmMsg("Start project");
		//Be careful with row/column major order!!! This will not work for current matrices.
		CLCore->Project(CatDocs, HierarchyLangId, TStrPr(HierarchyLangId, Lang2Id), PCatDocs);
		SwDispTmMsg("Finished project");
		//Be careful with row/column major order
		//TLinAlg::NormalizeData(PCatDocs);
		TLinAlg::NormalizeColumns(PCatDocs);
		SwDispTmMsg("Normalization complete");
	}	


	template <class val, class TSizeTy, bool colmajor>
	inline void TCLClassifierAbstract<val, TSizeTy, colmajor>::ComputeModel() {
		// TVec<TTriple<TInt, TIntV, TNum<val>> > NodeV; // (ParentNodeId, ChildNodeIdV, NodeVal)
		SwDispTmMsg("Start building tree");
		TCLCNodeAbstract< val, TSizeTy, colmajor> Root;//MODD
		Root.CatStr = "Top";
		Root.CatPathStr = "Top/";
		Taxonomy.AddRoot(Root);

		for (int DocN = 0; DocN < Cat.Len(); DocN++) {
			TStrV Path;
			Cat[DocN].SplitOnAllCh('/', Path, true);
			TStr CatPartial = "";
			int NodeId = 0;
			for (int PathNodeN = 0; PathNodeN < Path.Len(); PathNodeN++) {
				CatPartial += Path[PathNodeN] + "/";
				// find or create TCLCNode -> use/update: Taxonomy.GetNodeVal(NodeId).StrToChildIdH
				if (PathNodeN == 0) continue; //skip root 
				int KeyId = Taxonomy.GetNodeVal(NodeId).StrToChildIdH.GetKeyId(Path[PathNodeN]);
				if (KeyId >= 0) { // go in
					NodeId = Taxonomy.GetNodeVal(NodeId).StrToChildIdH[KeyId];					
				} else {
					// add
					TCLCNodeAbstract<val, TSizeTy, colmajor> NodeVal;
					NodeVal.CatPathStr = CatPartial;
					NodeVal.CatStr = Path[PathNodeN];	
					int ParentId = NodeId;
					NodeId = Taxonomy.AddNode(ParentId, NodeVal);
					// add to parent's StrToChildIdH: Path[PathNodeN], NodeId
					Taxonomy.GetNodeVal(ParentId).StrToChildIdH.AddDat(Path[PathNodeN], NodeId);
				}
				// update NodeId
				if (PathNodeN == Path.Len() -1) {
					// add DocN to TCLCNode
					//Taxonomy.GetNodeVal(NodeId).CatStr = Path[PathNodeN];
					Taxonomy.GetNodeVal(NodeId).DocIdxV.Add(DocN);
				}
			}			
		}
		SwDispTmMsg("Finished building tree");		
		TFOut TaxonomyWriter(ModelPath + "/Taxonomy.bin");
		Taxonomy.Save(TaxonomyWriter);			
		SwDispTmMsg("Finished saving tree with " + TInt::GetStr(Taxonomy.GetNodes()) + " nodes.");

		// Statistics: number of documents per cat
		//TreeStats();		
		//bfs
		TIntV IdxV;
		TreeBFS(Taxonomy, IdxV);
		//PrintTreeBFS(Taxonomy, IdxV, ModelPath + "/TreeBFS.txt");		
		IdxV.Reverse();
		//PrintTreeBFS(Taxonomy, IdxV, ModelPath + "/TreeReverseBFS.txt");

		// Centroid at node = mean( mean(data(node)), {centroids of children})
		Centroids.Gen(PCatDocs.GetRows(), Taxonomy.GetNodes());
		for (int CatN = 0; CatN < Taxonomy.GetNodes(); CatN++) {
			int NodeId = IdxV[CatN];
			//Andrej
			//index NodeId = IdxV[CatN];
			int Docs = Taxonomy.GetNodeVal(NodeId).DocIdxV.Len();
			int Rows = PCatDocs.GetRows();
			if (Docs > 0) {
				// internal centroid: sum
				for (int DocN = 0; DocN < Docs; DocN++) {
					//Fix this part works only for COLUMNS -> should be made independent
					// Storage order missmatch
					for (int RowN = 0; RowN < Rows; RowN++) {
						Centroids.At(RowN, NodeId) +=  PCatDocs.At(RowN, Taxonomy.GetNodeVal(NodeId).DocIdxV[DocN]);
					}
				}
				// internal centroid: average
				// Fix this part works only for COLUMNS -> should be made independent
				//  Storage order missmatch
				TLinAlg::NormalizeColumn(Centroids, NodeId);
			}
			/*for (int RowN = 0; RowN < Rows; RowN++) {
			Centroids.At(RowN, NodeId) /= (double)Docs;
			}*/
			// average internal centroid and centroids of children
			int Children = Taxonomy.GetChildren(NodeId);
			if (Children == 0) continue;
			// sum and normalize
			for (int ChildN = 0; ChildN < Children; ChildN++) {
				int ChildId = Taxonomy.GetChildNodeId(NodeId, ChildN);
				for (int RowN = 0; RowN < Rows; RowN++) {
					Centroids.At(RowN, NodeId) += Centroids.At(RowN, ChildId);
				}
			}
			TLinAlg::NormalizeColumn(Centroids, NodeId);
		}
		SwDispTmMsg("Finished computing centroids");		
		TFOut CentroidWriter(ModelPath + "/Centroids.bin");
		Centroids.Save(CentroidWriter);
		SwDispTmMsg("Finished saving (binary) centroids");	
			
	}

	template <class val, class TSizeTy, bool colmajor>
	inline void TCLClassifierAbstract<val, TSizeTy, colmajor>::Classify(TUStr& Text, TInt& Class, const TStr& TextLangId) {
		TPair<TIntV, TFltV> SparseVec;
		CLCore->TextToVector(Text, TextLangId, SparseVec);
		TFltV ProjVec;
		CLCore->Project(SparseVec, TextLangId, TStrPr(this->HierarchyLangId, this->Lang2Id), ProjVec, false);		
		ClassifyProjected(ProjVec, Class);
	}

	// Classify a single document, return top k classes
	template <class val, class TSizeTy, bool colmajor>
	inline void TCLClassifierAbstract<val, TSizeTy, colmajor>::Classify(TUStr& Text, TIntV& Class, const TStr& TextLangId, const int& k, const TBool& DmozSpecial) {
		//int kclip = TMath::Mn(k, Centroids.GetCols());
		//Andrej
		int kclip = MIN(k, Centroids.GetCols());
		TPair<TIntV, TFltV> SparseVec;
		CLCore->TextToVector(Text, TextLangId, SparseVec);
		if (SparseVec.Val1.Empty()) {
			return;
		}
		TFltV ProjVec;
		CLCore->Project(SparseVec, TextLangId, TStrPr(this->HierarchyLangId, this->Lang2Id), ProjVec, DmozSpecial);
		ClassifyProjected(ProjVec, Class, kclip);
	}

	//Fix, this part needs templatization
	template <class val, class TSizeTy, bool colmajor>
	inline void TCLClassifierAbstract<val, TSizeTy, colmajor>::ClassifyProjected(const TFltV& ProjVec, TInt& Class) {
		//SwDispTmMsg("Start classification");
		TFltV Similarities;
		Similarities.Gen(Centroids.GetCols());
		//SwDispTmMsg("Start multiply: Centroids' * ProjVec, dims = " + TInt::GetStr(Centroids.GetRows()) + " " + TInt::GetStr(Centroids.GetCols()) + " " + TInt::GetStr(ProjVec.Len()));
		TLinAlg::MultiplyT(Centroids, ProjVec, Similarities);
		Class = Similarities.GetMxValN();
		//SwDispTmMsg("Finished classification");
	}	
	//Add TFltVV ProjVecVV
	template <class val, class TSizeTy, bool colmajor>
	inline void TCLClassifierAbstract<val, TSizeTy, colmajor>::ClassifyProjected(const TFltV& ProjVec, TIntV& Class, const int& k) {
		//Andrej
		//int kclip = TMath::Mn(k, Centroids.GetCols()); 
		int kclip = MIN(k, Centroids.GetCols()); 
		//SwDispTmMsg("Start classification");
		TFltV Similarities;
		Similarities.Gen(Centroids.GetCols());
		// return index of maximum Centroids' * ProjVec
		//SwDispTmMsg("Start multiply: Centroids' * ProjVec, dims = " + TInt::GetStr(Centroids.GetRows()) + " " + TInt::GetStr(Centroids.GetCols()) + " " + TInt::GetStr(ProjVec.Len()));
		TLinAlg::MultiplyT(Centroids, ProjVec, Similarities);
		TVec<TPair<TFlt,TInt> > AugSimilarities(Similarities.Len());
		for (int ElN = 0; ElN < Similarities.Len(); ElN++) {
			AugSimilarities[ElN].Val1 = Similarities[ElN];
			AugSimilarities[ElN].Val2 = ElN;
		}
		AugSimilarities.Sort(false);
		Class.Gen(kclip);
		for (int ElN = 0; ElN < kclip ; ElN++) {
			Class[ElN] = AugSimilarities[ElN].Val2;
		}
		//SwDispTmMsg("Finished classification");
	}

	template <class val, class TSizeTy, bool colmajor>
	inline void TCLClassifierAbstract<val, TSizeTy, colmajor>::ClassifyProjected(const TFltVV& ProjVec, TVec<TIntV>& ClassVV, const int& k) {
		int kclip = TMath::Mn(k, Centroids.GetCols());
		//SwDispTmMsg("Start classification");
		TFltVV Similarities;
		Similarities.Gen(ProjVec.GetRows(), Centroids.GetCols());
		// return index of maximum Centroids' * ProjVec
		//SwDispTmMsg("Start multiply: Centroids' * ProjVec, dims = " + TInt::GetStr(Centroids.GetRows()) + " " + TInt::GetStr(Centroids.GetCols()) + " " + TInt::GetStr(ProjVec.Len()));
		TLinAlg::Multiply(ProjVec, Centroids, Similarities);
		TVVec<TPair<TFlt, TInt> > AugSimilarities(Similarities.GetXDim(), Similarities.GetYDim());
		for (int i = 0; i < Similarities.GetXDim(); i++){
			TVec<TPair<TFlt, TInt> > SimPair; AugSimilarities.GetRowPtr(i, SimPair);
			TFltV SimV; Similarities.GetRowPtr(i, SimV);

			for (int ElN = 0; ElN < SimV.Len(); ElN++) {
				SimPair[ElN].Val1 = SimV[ElN];
				SimPair[ElN].Val2 = ElN;
			}
		}
#pragma omp parallel for
		for (int i = 0; i < Similarities.GetXDim(); i++){
			TVec<TPair<TFlt, TInt> > SimPair; AugSimilarities.GetRowPtr(i, SimPair);
			SimPair.Sort(false);
		}
		ClassVV.Gen(Similarities.GetXDim(), Similarities.GetXDim());
#pragma omp parallel for
		for (int i = 0; i < Similarities.GetXDim(); i++){
			TIntV& ClassV = ClassVV[i];
			ClassV.Gen(kclip);
			TVec<TPair<TFlt, TInt> > SimPair; AugSimilarities.GetRowPtr(i, SimPair);
			for (int ElN = 0; ElN < kclip; ElN++) {
				ClassV[ElN] = SimPair[ElN].Val2;
			}
		}
		//SwDispTmMsg("Finished classification");
	}

	template <class val, class TSizeTy, bool colmajor>
	inline void TCLClassifierAbstract<val, TSizeTy, colmajor>::GetBestKWordV(const TStrV& PathV, TStrV& KeywordV, TIntV& KeyFqV) {
		THash<TStr, TInt> KwH;
		for (int PathN = 0; PathN < PathV.Len(); PathN++) {
			TStrV Parts;
			PathV[PathN].SplitOnAllCh('/', Parts, true);
			for (int PartN = 0; PartN < Parts.Len(); PartN++) {
				if (KwH.IsKey(Parts[PartN])) {
					KwH.GetDat(Parts[PartN])++;
				}
				else {
					KwH.AddDat(Parts[PartN], 1);
				}
			}
		}
		KwH.SortByDat(false);
		KwH.GetKeyV(KeywordV);
		KwH.GetDatV(KeyFqV);
	}

	template <class val, class TSizeTy, bool colmajor>
	inline void TCLClassifierAbstract<val, TSizeTy, colmajor>::GetBestKWordV(const TIntV& ClassV, TStrV& KeywordV, TIntV& KeyFqV) {
		TStrV PathV;
		for (int ElN = 0; ElN < ClassV.Len(); ElN++) {
			TStr Path = GetClassPathStr(ClassV[ElN]);
			TStrV Words; Path.SplitOnAllCh('_', Words, true);
			for (int WordN = 0; WordN < Words.Len(); WordN++) {
				PathV.Add(Words[WordN].ToLc());
			}
		}
		GetBestKWordV(PathV, KeywordV, KeyFqV);
	}

	template <class val, class TSizeTy, bool colmajor>
	inline void TCLClassifierAbstract<val, TSizeTy, colmajor>::GetBestKWordV(const TIntV& ClassV, TStrV& KeywordV, TIntV& KeyFqV, TStrV& Categories) {
		TStrV PathV;
		for (int ElN = 0; ElN < ClassV.Len(); ElN++) {
			TStr Path = GetClassPathStr(ClassV[ElN]);
			Categories.Add(Path);
			TStrV Words; Path.SplitOnAllCh('_', Words, true);
			for (int WordN = 0; WordN < Words.Len(); WordN++) {
				PathV.Add(Words[WordN].ToLc());
			}
		}
		GetBestKWordV(PathV, KeywordV, KeyFqV);
	}

	template <class val, class TSizeTy, bool colmajor>
	inline void TCLClassifierAbstract<val, TSizeTy, colmajor>::GetBestKWordV(const TVec<TIntV> & ClassVV, TVec<TStrV>& KeywordVV, TVec<TIntV> & KeyFqVV, TVec<TStrV>& CategoriesVV) {
		int n_samples = ClassVV.Len();
		if (KeywordVV.Empty() || KeywordVV.Len() != n_samples){
			KeywordVV.Gen(n_samples, n_samples);
		}
		if (KeyFqVV.Empty() || KeyFqVV.Len() != n_samples){
			KeyFqVV.Gen(n_samples, n_samples);
		}
		if (CategoriesVV.Empty() || CategoriesVV.Len() != n_samples){
			CategoriesVV.Gen(n_samples, n_samples);
		}

#pragma omp parallel for
		for (int i = 0; i < ClassVV.Len(); i++){
			const TIntV& ClassV = ClassVV[i];
			TStrV& KeywordV = KeywordVV[i];
			TIntV& KeyFqV = KeyFqVV[i];
			TStrV& CategoriesV = CategoriesVV[i];
			GetBestKWordV(ClassV, KeywordV, KeyFqV, CategoriesV);
		}
	}

	template <class val, class TSizeTy, bool colmajor>
	inline void TCLClassifierAbstract<val, TSizeTy, colmajor>::PrintTreeBFS(TTree<TCLCNodeAbstract<val, TSizeTy, colmajor> >& Tree, const TIntV& IdxV, const TStr& FNm) {
		TFOut FOut(FNm);
		for (int NodeN = 0; NodeN < IdxV.Len(); NodeN++) {	
			/*if (NodeN < 30) {
			Tree.GetNodeVal(IdxV[NodeN]).Disp();
			}*/
			FOut.PutStr(Tree.GetNodeVal(IdxV[NodeN]).CatPathStr + "\n");
		}
	}
	

}
#endif