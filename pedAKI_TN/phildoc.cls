\ProvidesClass{phildoc}
% make sure that, whatever is defined in texmf.cnf, the output will be pdf:
\pdfoutput=1

\parindent=1em
\parskip=0ex

\DeclareOption*{\PassOptionsToClass{\CurrentOption}{memoir}}
\ProcessOptions
\LoadClass{memoir}

\newdimen\wtrimmed
\newdimen\htrimmed
\newdimen\wstock
\newdimen\hstock

% images are in subdirectory graphics at their final resolution, 
% in graphics_lowres keep the same images with low resolution, used during development
% the option lowres selects the low resolution images
\newif\iflowres\lowresfalse\DeclareOption{lowres}{\lowrestrue}

% options to make the difference between a thesis, a technical note and stellingen:
\newif\ifthesis\thesisfalse\DeclareOption{thesis}{\thesistrue}
\newif\iftechnote\technotefalse\DeclareOption{technote}{\technotetrue}
\newif\ifstellingen\stellingenfalse\DeclareOption{stellingen}{\stellingentrue}

\DeclareOption{a4}{\wstock=210mm\hstock=297mm}
\DeclareOption{letter}{\wstock=216mm\hstock=279mm}
\DeclareOption{160x240}{\wtrimmed=160mm\htrimmed=240mm}
\DeclareOption{190x250}{\wtrimmed=190mm\htrimmed=250mm}

\newif\iftestlanguageoptionspecified

\DeclareOption{english}{\testlanguageoptionspecifiedtrue
  \PassOptionsToPackage{english}{babel}}
\DeclareOption{dutch}{\testlanguageoptionspecifiedtrue
  \PassOptionsToPackage{dutch}{babel}}
\ExecuteOptions{a4,160x240}
\ProcessOptions
\iftestlanguageoptionspecified\else
  \PassOptionsToPackage{dutch,english}{babel}\fi

%\RequirePackage{listings}
\RequirePackage{calc}			% dimensions can be added, multiplied et cetera
\RequirePackage{colortbl}		% for tables with colored cells
\RequirePackage{xspace}			% lets you define commands that won't need \ at the end
\RequirePackage[authoryear,round]{natbib}% bibliography package
\bibliographystyle{plainnat}
\RequirePackage{amsmath}
\RequirePackage{txfonts}		% use txfonts with old style figures 
\RequirePackage{longtable}		% for longtables
\RequirePackage{ctable}			% for nicer tabel and figure floats
\RequirePackage{babel}			% language support, default is english
\RequirePackage{graphicx}		% for graphics...
\iflowres				% if the lowres option was set,
\graphicspath{{graphics_lowres/}}	% look for graphics in the directory graphics_lowres
\else					% else
\graphicspath{{graphics/}}		% look in graphics
\fi
\RequirePackage{varioref}		% for fuzzy referencing

% page setup
\def\bindingoffset{0mm}              	% set only \ifshowtrims
\setstocksize{\hstock}{\wstock}          	% print on A4 paper
\settrimmedsize{\hstock}{\wstock}{*}       	% print on A4 paper
\iftechnote\else			% no trimming or showtrims for technotes (alwas A4)
   \ifshowtrims
      \def\bindingoffset{8mm}           % this disappears in the binding
   \else
      \setstocksize{\htrimmed}{\wtrimmed}       % print on ready-cut paper
   \fi
   \settrimmedsize{\htrimmed}{\wtrimmed}{*} 	% royal format after cutting
\fi
\setlength{\trimtop}{0pt}		% top doesn't need cutting
\setlength{\trimedge}{\stockwidth-\paperwidth-\bindingoffset}

% put a 9x9 grid over the page
\newlength{\hgrid}\setlength{\hgrid}{\paperheight/9}
\newlength{\wgrid}\setlength{\wgrid}{\paperwidth/9}

% the typeblock is 7/9 in both directions
\settypeblocksize{7\hgrid}{7\wgrid}{*}%

% with 1/9 margins
\setulmargins{\hgrid}{*}{*}
\setlrmargins{\wgrid}{*}{*}

\setmarginnotes{17pt}{51pt}{\onelineskip}
\setheadfoot{\onelineskip}{2\onelineskip}
\setheaderspaces{*}{\onelineskip}{*}
\checkandfixthelayout
\typeoutlayout
\typeoutstandardlayout

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%    General commands     %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

\def\sfbold#1{\textsf{\textbf{\small#1}}} % bold sans serif
\def\itbold#1{\textit{\textbf{#1}}}	  % bold italic
\def\itbold#1{\textsl{\textbf{#1}}}	  % bold slanted
\def\think{\leavevmode%
  \unskip\thinspace\penalty0\hbox{--\kern-.25em--}\thinspace\penalty0\ \hskip-.2em
}

\newcommand{\attrib}[1]{\hspace*{\fill}{\footnotesize #1}}

\def\COPYRIGHTnote{%
   {\small\textcopyright}
   \textsc{\small koninklijke philips electronics nv \the\year}
}

% \Index{something} prints something and makes an index entry for
% it. In the index, something's alphabetical position is determined
% after removal of any characters other than [a-zA-Z0-9] and after
% making lower case.

\def\wybostopwoord{wybostopwoord}

\def\doconvertargument#1>{}

\def\nofinalspace#1 \wybostopwoord{#1}

\def\makeascii#1{%
  \long\def\ascii{#1}%
  \edef\text{\expandafter\doconvertargument\meaning\ascii\space wybostopwoord }%
  \edef\ascii{\expandafter\runtrsp\text}%
  \edef\ascii{\expandafter\nofinalspace\ascii\wybostopwoord}}

\def\runtrsp#1 {%
  \expandafter\ifx\csname #1\endcsname\wybostopwoord
  \else 
    \runtr#1\relax \space
    \expandafter \runtrsp
 \fi 
}

\def\runtr#1%
  {\ifx#1\relax\else
   \ifcase`#1
     \or  \or  \or  \or  \or  \or  \or  \or  \or  \or  \or  \or  \or  \or  \or  \or
     \or  \or  \or  \or  \or  \or  \or  \or  \or  \or  \or  \or  \or  \or  \or  \or
     \or  \or  \or  \or  \or  \or  \or  \or  \or  \or  \or  \or  \or  \or  \or  \or
    0\or 1\or 2\or 3\or 4\or 5\or 6\or 7\or 8\or 9\or  \or  \or  \or  \or  \or  \or
     \or a\or b\or c\or d\or e\or f\or g\or h\or i\or j\or k\or l\or m\or n\or o\or
    p\or q\or r\or s\or t\or u\or v\or w\or x\or y\or z\or  \or  \or  \or  \or  \or
     \or a\or b\or c\or d\or e\or f\or g\or h\or i\or j\or k\or l\or m\or n\or o\or
    p\or q\or r\or s\or t\or u\or v\or w\or x\or y\or z\or  \or  \or  \or  \or  \or
     \or  \or  \or  \or  \or  \or  \or  \or  \or  \or  \or  \or  \or  \or  \or  \or
     \or  \or  \or  \or  \or  \or  \or  \or  \or  \or  \or  \or  \or  \or  \or  \or
     \or  \or  \or  \or  \or  \or  \or  \or  \or  \or  \or  \or  \or  \or  \or  \or
     \or  \or  \or  \or  \or  \or  \or  \or  \or  \or  \or  \or  \or  \or  \or  \or
     \or  \or  \or  \or  \or  \or  \or  \or  \or  \or  \or  \or  \or  \or  \or  \or
     \or  \or  \or  \or  \or  \or  \or  \or  \or  \or  \or  \or  \or  \or  \or  \or
     \or  \or  \or  \or  \or  \or  \or  \or  \or  \or  \or  \or  \or  \or  \or  \or
     \or  \or  \or  \or  \or  \or  \or  \or  \or  \or  \or  \or  \or  \or  \or  \or
  \fi
  \expandafter\runtr\fi}

% \Index[sub]{\LaTeX} prints \LaTeX and indexes it under latex
% if sub is given it is index as item sub under \LaTeX
% so \Index[sub]{\LaTeX} is equivalent to \LaTeX\index{latex@\LaTeX!sub}

\newcommand{\Index}[2][]{\makeascii{#2}\index{\ascii!#1}#2}

% use tt from computer modern
\renewcommand{\ttdefault}{cmtt}

% We like a smaller verbatim, but not in running text.
% \every@verbatim doesn't apply to the inline verb environment.

\def\sm@ller{%
  \@tempdima=\f@size pt
  \advance\@tempdima-0.1\@tempdima
  %\@tempdimb=\f@baselineskip
  %\advance\@tempdimb-0.1\@tempdimb
  \fontsize{\@tempdima}{\f@baselineskip}\selectfont}
\addto@hook\every@verbatim{\sm@ller}%

% less whitespace around verbatims
\def\less@white{\topsep=3pt}
\addto@hook\every@verbatim{\less@white}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%    Thesis commands      %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

\newcommand{\halftitlerecto}{%
   % halftitlepage recto
   \pagestyle{empty}
   \begin{center}
      {\Large\scshape\Maintitle}
   \end{center}
   \vfill
   \ifx\Coverlegend\empty\else\noindent Cover:\\\Coverlegend\fi
   \newpage
}

\newcommand{\halftitleverso}{%
   % halftitlepage verso
   \noindent The work described in this thesis has been carried out at the\\
   \textsc{philips research laboratories}, Eindhoven, The Netherlands,\\
   as part of the Philips Research Programme.\\[2ex]
   \COPYRIGHTnote\\
   All rights are reserved; reproduction in whole or in part is\\
   prohibited without the written consent of the copyright owner.
   \ifx\Isbn\empty\else
      \vfill
      \textsc{isbn} \Isbn
   \fi
   \newpage
}

\newcommand{\titlerecto}{%
   \vspace*{\fill}
   \begin{center}
      {\huge\Maintitle\mbox{}}
      \ifx\empty\else\\[2ex]{\large\Subtitle\mbox}{}\fi
      \\[5ex]\textsc{\LARGE proefschrift}\\[5ex]
      ter verkrijging van de graad van doctor\\
      aan de \University,\\
      op gezag van de Rector Magnificus, \Rectorname,\\
      in het openbaar te verdedigen ten overstaan van een commissie,\\
      door het College van Dekanen aangewezen,\\
      op \Promodate, te \Promotime\ uur\\
      door\\[3ex]
      {\Large\Fullname}\\[3ex]
      geboren \Borndate\ te \Bornplace
   \end{center}
   \vfill\vfill
   \newpage
}

\newcommand{\titleverso}{%
   \noindent Dit proefschrift is goedgekeurd door de promotor:\\[2ex]
   \Promotorname\\[2ex]
   Samenstelling promotiecommissie:\\[2ex]
   \Rectorname, Rector Magnificus, voorzitter\\
   \Promotorname, promotor\\
   \newpage
}

\newcommand{\dedicationsheet}{%
   \ifx\Dedication\empty\else
      \vspace*{\fill}
      {\flushright\itshape\Dedication\\\mbox{}}
      \cleardoublepage
   \fi
}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%   Stellingen commands   %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

\newcommand{\frontpage}[2]{
   \vspace*{\fill}
   \begin{center}
   \textbf{\LARGE STELLINGEN}\\[5ex]
   behorende bij het proefschrift\\[5ex]
   {\Large\Maintitle}\\[5ex]
   door \Authorname\par
   \end{center}\par
   \vfill\vfill
   \cleardoublepage
}

\newenvironment{stellingen}
   {\begin{adjustwidth}{0mm}{10mm}\begin{enumerate}}
   {\end{enumerate}\end{adjustwidth}}
\pagestyle{empty}

\newcommand{\stelling}[2][]{
   \def\attribute{#1}
   \item #2\ifx\empty\attribute\else\\\attrib{#1}\fi
}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Technical Note commands %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% \Restriction:
% 0: Unclassified
% 1: Philips restricted
% 2: Company Confidential
% 3: Philips/IMEC restricted
% 4: Crolles2Alliance restricted

\def\Noteid{PR-TN-\Noteyr/\Noteno}
\def\Restrictionheader{%
  \ifcase\Restriction
    Unclassified\or
    Philips Restricted\or
    Company Confidential until \Mcfunt/\Ycfunt\or
    Philips/IMEC restricted\or
    Crolles2Alliance restricted
  \fi
}

\def\CONFItext{%
 \ifcase\Restriction
   \Restrictionheader\or
   \Restrictionheader\or
   \fboxsep=2ex
   {\LARGE\bfseries\Restrictionheader}\\[2ex]
   \fbox{\begin{minipage}{118mm}\small
   Company Confidential reports are issued personally. The receiver of
   such a report must ensure that this information is not shared with
   anyone unauthorized, inside or outside Philips. Access by others has
   to be approved individually by the group leader of the first author.
   \end{minipage}}\\[2ex]
   \COPYRIGHTnote\or
   \Restrictionheader\or
   \Restrictionheader
 \fi
}

\def\COPYRIGHTtext{%
   \fboxsep=2ex
   \fbox{%
      \begin{minipage}{118mm}
         \COPYRIGHTnote\\[2ex]
         All rights reserved. Reproduction or dissemination in whole or in
         part is prohibited without the prior written consent of the
         copyright holder.
      \end{minipage}
   }
}

\def\Tnotefrontmatter{
   \setsecnumdepth{paragraph}
   \maxsecnumdepth{paragraph}
   \settocdepth{paragraph}
   \maxtocdepth{paragraph}
   \pagenumbering{roman}
   
   % Change the name of the bibliography to "References".
   \def\bibname{\iflanguage{dutch}{Referenties}{References}}
   \thispagestyle{empty}
   
   \vspace*{20mm}
   \hspace*{20mm}%
   \begin{minipage}[t][172mm][t]{125mm}
      \Large
      Technical Note \Noteid\\[21mm]
      Issued: \Miss/\Yiss\\[26mm]
      {\LARGE\bfseries \Maintitle}\\[24mm]
      \Authorname\\[4mm]
      \Audept
      \normalsize
      \vfill
      \CONFItext
   \end{minipage}
   \newpage\vspace*{130mm}

   \makepagestyle{tnote}
   \makeevenhead{tnote}{\Noteid}{}{\Restrictionheader}
   \makeoddhead{tnote}{\Restrictionheader}{}{\Noteid}
   \makeevenfoot{tnote}{\thepage}{}{\COPYRIGHTnote}
   \makeoddfoot{tnote}{\COPYRIGHTnote}{}{\thepage}

   \makepagestyle{titleverso}
   \makeevenhead{titleverso}{\Noteid}{}{\Restrictionheader}
   \makeevenfoot{titleverso}{\thepage}{}{}

   \aliaspagestyle{chapter}{tnote}

   \pagestyle{titleverso}
   \begin{tabular}{@{}ll@{}}
      Author's address:&\Authorname\\
                       &\Authormail
   \end{tabular}
   \vfill\COPYRIGHTtext
   \newpage
   \pagestyle{tnote}
   \begin{tabularx}{\hsize}{@{}>{\bfseries}rX@{}}
      Title:          &\Maintitle\\\\
      Author(s):      &\Authorname\\\\
      Reviewer(s):    &\Reviewernames\\\\
      Technical Note: &\Noteid\\\\
      Additional      &\Additionalnos\\
      numbers:        &\\\\
      Subcategory:    &\Subcategory\\\\
      Project:        &\Project\\\\
      Customer:       &\Customer\\\\
      Keywords:       &\Keywords\\\\
      Abstract:       &\Abstract\\\\
      Conclusions:    &\Conclusion\\
   \end{tabularx}
   \ifdefined\MngmntSum
     \cleardoublepage
     \chapter*{Management summary}
       \MngmntSum
   \fi
   \cleardoublepage
   \tableofcontents
   \cleardoublepage
   \pagenumbering{arabic}\setcounter{page}{1}
}

\endinput

$Id: phildoc.cls,v 1.3 2005/09/03 11:07:15 wybo Exp $
vim: syn=tex
