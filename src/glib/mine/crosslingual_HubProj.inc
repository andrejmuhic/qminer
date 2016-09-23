/**
* Copyright (c) 2015, Jozef Stefan Institute, Andrej Muhic
* All rights reserved.
*
* This source code is licensed under the FreeBSD license found in the
* LICENSE file in the root directory of this source tree.
*/
	template <class val, class TSizeTy, bool colmajor>
	void TCLHubProjectorAbstract<val, TSizeTy, colmajor>::Project(const TPair<TVec<TNum<TSizeTy>, TSizeTy>, TVec<TNum<val>, TSizeTy>>& Doc, const TStr& DocLangId, const TStrPr& TargetSpace, TVec<TNum<val>, TSizeTy>& Projected, const TBool& DmozSpecial) const{
		const TVVec<TNum<val>, TSizeTy, colmajor> * ProjMat = NULL;
		const TVec<TNum<val>, TSizeTy>* Center = NULL;
		const TVVec<TNum<val>, TSizeTy, colmajor>* ConMat = NULL;
		const TVec<TNum<val>, TSizeTy>* InvDocV = NULL;

		TBool transpose;
		bool OK = TCLHubProjector::SelectMatrices(DocLangId, TargetSpace, ProjMat, Center, ConMat, InvDocV, transpose);
		
		//Do not go forward if data was not initialized
		if (!OK) return;
		if (ConMat != NULL){}

		// Reweight Doc!
		TPair<TVec<TNum<TSizeTy>, TSizeTy>, TVec<TFlt, TSizeTy>> DocTemp = Doc;
		//val norm = 0.0;
		for (TSizeTy ElN = 0; ElN < Doc.Val1.Len(); ElN++) {
			DocTemp.Val2[ElN] = Doc.Val2[ElN] * (*InvDocV)[Doc.Val1[ElN]];
			//norm = norm + DocTemp.Val2[ElN] * DocTemp.Val2[ElN];
		}
		//norm = 1.0/sqrt(norm);

		//Projected = ConMat(ProjMat*Doc - Center) ... disabled centering
		TVec<TNum<val>, TSizeTy> Pd;
		Pd.Gen( colmajor ? ProjMat->GetRows() : ProjMat->GetCols() );

		colmajor ? TLinAlg::Multiply(*ProjMat, DocTemp, Pd) : TLinAlg::MultiplyT(DocTemp, *ProjMat, Pd);

#ifdef DEBUG_ENABLE
		printf("Projection done!\n");
#endif DEBUG_ENABLE
		/*TLinAlg::MultiplyScalar(norm, Pd);
		TFltV Pd_minus_c(Center->Len());*/
		//[Andrej]
		//TLinAlg::AddVec(0.0, *Center, Pd, Pd_minus_c);
#ifdef DEBUG_ENABLE
		printf("First component of center: %g\n", (*Center)[0].Val);
		printf("Norm of center: %g, Norm of projdoc: %g\n", TLinAlg::Norm(*Center), TLinAlg::Norm(Pd));
		printf("Pd_minus done! Center: %d; Pd: %d; Pd_minus_c: %d\n", Center->Len(), Pd.Len(), Pd_minus_c.Len());
#endif

		if ( (!DmozSpecial && Projected.Empty()) ) {
			//printf("Generation of projected vector of length %d \n", ConMat->GetRows());
			//[Andrej]
		#ifdef DEBUG_ENABLE
			printf("Generation of projected vector started\n");
		#endif
			//Projected.Gen(ConMat->GetRows());
			colmajor ? Projected.Gen(ConMat->GetRows()) : Projected.Gen(ConMat->GetCols());
		#ifdef DEBUG_ENABLE
			printf("Generation of projected vector done\n");
		#endif
		}
		//When DmozSpecial flag is enabled identity matrix is assumed ... no change!!!
		if (!DmozSpecial){
		#ifdef DEBUG_ENABLE
			printf("Normal multiplication!\n");
		#endif
			colmajor ? TLinAlg::Multiply(*ConMat, Pd, Projected, transpose) : TLinAlg::Multiply(*ConMat, Pd, Projected, transpose);
		}
		else{
			//Projected = std::move(Pd_minus_c);
		}
	}

	template <class val, class TSizeTy, bool colmajor>
	void TCLHubProjectorAbstract<val, TSizeTy, colmajor>::Project(const TTriple<TVec<TNum<TSizeTy>, TSizeTy>, TVec<TNum<TSizeTy>, TSizeTy>, TVec<TNum<val>, TSizeTy>>& DocMatrix, const TStr& DocLangId, const TStrPr& TargetSpace, TVVec<TNum<val>, TSizeTy, colmajor>& Projected, const TBool& Tfidf, const TBool& DmozSpecial) const {
		const TVVec<TNum<val>, TSizeTy, colmajor>* ProjMat = NULL;
		const TVec<TNum<val>, TSizeTy>* Center = NULL;
		const TVVec<TNum<val>, TSizeTy, colmajor>* ConMat = NULL;
		const TVec<TNum<val>, TSizeTy>* InvDocV = NULL;
		//printf("start select\n");
		//			bool OK = TCLHubProjector::SelectMatrices(DocLangId, Lang2Id, ProjMat, Center, ConMat, InvDocV, DmozSpecial);
		//			TLinAlg::MultiplyT(*ConMat, Pd_minus_c, Projected);
		TBool transpose;
		bool OK = SelectMatrices(DocLangId, TargetSpace, ProjMat, Center, ConMat, InvDocV, transpose);
		//Do not go forward if data was not initialized
		if (!OK) return;

		if (!OK){
			if (DmozSpecial){
				//printf("Dmoz special case, identity matrix assumed!\n");
			}
			else{
				//printf("select failed\n");
			}
		}
		if (OK){
			//printf("selected hub projector %s %s, dims %d %d %d %d %d %d\n", DocLangId.CStr(), Lang2Id.CStr(), Center->Len(), ConMat->GetRows(), ConMat->GetCols(), InvDocV->Len(), ProjMat->GetRows(), ProjMat->GetCols());
		}
		else{
			//printf("selected hub projector %s %s identity assumed, dims %d %d %d %d %d %d\n", DocLangId.CStr(), Lang2Id.CStr(), Center->Len(), Center->Len(), Center->Len(), InvDocV->Len(), ProjMat->GetRows(), ProjMat->GetCols());
		}

		TTmStopWatch time;
		TTriple<TVec<TNum<TSizeTy>, TSizeTy>, TVec<TNum<TSizeTy>, TSizeTy>, TVec<TNum<val>, TSizeTy>> DocMatrixTemp;
		if (Tfidf){
			DocMatrixTemp = DocMatrix;
			//time.Start();
			for (TSizeTy ElN = 0; ElN < DocMatrixTemp.Val1.Len(); ElN++) {
				DocMatrixTemp.Val3[ElN] = DocMatrix.Val3[ElN] * (*InvDocV)[DocMatrix.Val1[ElN]];
			}
			//time.Stop("Inverse indexing costs: ");
			//time.Reset(false);
		}
		//Projected = ConMat(ProjMat*DocMatrix - Center* ones(1, size(DocMatrix,2)))
		TSizeTy Docs = DocMatrix.Val2[DocMatrix.Val2.GetMxValN()] + 1;
		//printf("Number of docs %d\n", Docs);
		// Compute: Pd_minus_cones := ProjMat*DocMatrix - Center* ones(1, size(DocMatrix,2))
		TVVec<TNum<val>, TSizeTy, colmajor> Pd_minus_cones;
#ifdef COLMAJOR_DATA
		Pd_minus_cones.Gen(ProjMat->GetRows(), Docs);
#else
		Pd_minus_cones.Gen(Docs, ProjMat->GetCols());
#endif
		//printf("Vector generated \n");
		//printf("Start multiply sparse %d", Docs);
		//time.Start();
		//ROW and COLUMN major mess
		//TLinAlg::Transpose(DocMatrixTemp, DocMatrixTempT);
		if (Tfidf){
			//printf("TFidf\n");
#ifdef COLMAJOR_DATA
			//Andrej index template madness, add IndexType to template
			TLinAlg::Multiply(*ProjMat, DocMatrixTemp, Pd_minus_cones);
#else
			TLinAlg::MultiplyT(DocMatrixTemp, *ProjMat, Pd_minus_cones);

			//Andrej index template madness, add IndexType to template
#endif
		}
		else{
#ifdef COLMAJOR_DATA
			//Andrej index template madness, add IndexType to template
			TLinAlg::Multiply(*ProjMat, DocMatrix, Pd_minus_cones);
#else
			//Andrej index template madness, add IndexType to template
			TLinAlg::MultiplyT(DocMatrix, *ProjMat, Pd_minus_cones);
#endif
		}
		//time.Stop("Projection costs: ");
		//time.Reset(false);
		//SwDispTmMsg("finished multiply");
		//time.Start();

#ifdef COLMAJOR_DATA
#ifndef INTEL
		for (int ColN = 0; ColN < Pd_minus_cones.GetCols(); ColN++) {
			TLinAlg::AddVec(-1.0, *Center, Pd_minus_cones, ColN);
		}
#else
		TFltV enke(Pd_minus_cones.GetCols()); TLAMisc::Fill(enke, 1);
		for (int RowN = 0; RowN < Pd_minus_cones.GetRows(); RowN++) {
			TFltV row; Pd_minus_cones.GetRowPtr(RowN, row);
			TLinAlg::AddVec((*Center)[RowN].Val, enke, row);
		}
#endif
#else
#ifndef INTEL
		for (int RowN = 0; RowN < Pd_minus_cones.GetRows(); RowN++) {
			TFltV row;
			Pd_minus_cones.GetRowPtr(RowN, row);
			TLinAlg::AddVec(-1.0, *Center, row, row);
			//TLinAlg::AddVec(-1.0, *Center, row);
			//TLinAlg::AddVec(-1.0, *Center, Pd_minus_cones, RowN);
		}
#else
		/*Andrej HOOT
		for (TSizeTy RowN = 0; RowN < Pd_minus_cones.GetRows(); RowN++) {
			//printf("%d/%d\n", RowN, Pd_minus_cones.GetRows());
			TVec<TVal, TSizeTy> row;
			Pd_minus_cones.GetRowPtr(RowN, row);
			TLinAlg::AddVec(TVal(-1.0).Val, *Center, row);
		}
		*/
#endif
#endif

		//time.Stop("Naive centering of data: ");
		//time.Reset(false);
		//SwDispTmMsg("finished advec");
		if (Projected.Empty() && !DmozSpecial) {
			//printf("Projected is empty!!!");
#ifdef COLMAJOR_DATA
			Projected.Gen(ConMat->GetRows(), Pd_minus_cones.GetCols());
#else
			Projected.Gen(Pd_minus_cones.GetRows(), ConMat->GetCols());
#endif
		}
		//SwDispTmMsg("finished gen");
		//time.Start();
		if (ConMat != NULL){
			if ( (DmozSpecial || (*ConMat)(0, 0).Val == 1.0) ){
				//[ANDREJ]printf("Identity matrix detected!!!\n");
				Projected = Pd_minus_cones;
			}			//			colmajor ? TLinAlg::Multiply(*ConMat, Pd_minus_c, Projected, transpose) : TLinAlg::Multiply(*ConMat, Pd_minus_c, Projected, transpose);


			else{
#ifdef COLMAJOR_DATA
				TLinAlg::Multiply(*ConMat, Pd_minus_cones, Projected);
#else
				//			TLinAlg::MultiplyT(*ConMat, Pd_minus_c, Projected);
				TLinAlg::Multiply(Pd_minus_cones, *ConMat, Projected, false, !transpose);
#endif

			}
		}
		else{
			Projected = Pd_minus_cones;
		}
		//time.Stop("Projection with connection matrix costs: ");
		//SwDispTmMsg("finished multiply conmat pd_minus_cones");
	}