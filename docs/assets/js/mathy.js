/*
 * Mathy documentation JavaScript.
 *
 * Adds high-value usability features for long MkDocs/mkdocstrings API pages:
 * - Back-to-top button
 * - API expand/collapse controls
 * - API object filter
 * - Copy import path buttons
 * - Code block language badges
 * - Dark-mode image helper
 */
( function()
{
	"use strict";
	
	function onReady( callback )
	{
		if( document.readyState === "loading" )
		{
			document.addEventListener( "DOMContentLoaded", callback );
		}
		else
		{
			callback();
		}
	}
	
	function createElement( tag, options )
	{
		const element = document.createElement( tag );
		if( !options )
		{
			return element;
		}
		if( options.className )
		{
			element.className = options.className;
		}
		if( options.text )
		{
			element.textContent = options.text;
		}
		if( options.html )
		{
			element.innerHTML = options.html;
		}
		if( options.attributes )
		{
			Object.entries( options.attributes ).forEach( ( [ key, value ] ) =>
			{
				element.setAttribute( key, value );
			} );
		}
		return element;
	}
	
	function getMainContent()
	{
		return document.querySelector( ".md-content__inner" ) || document.querySelector( "main" );
	}
	
	function isApiPage()
	{
		return Boolean( document.querySelector( ".doc-object, .doc-heading, [id^='api-']" ) );
	}
	
	function addBackToTopButton()
	{
		if( document.querySelector( ".mathy-back-to-top" ) )
		{
			return;
		}
		const button = createElement( "button", {
			className: "mathy-back-to-top",
			text: "↑ Top",
			attributes: {
				type: "button",
				"aria-label": "Back to top"
			}
		} );
		button.addEventListener( "click", function()
		{
			window.scrollTo( {
				top: 0,
				behavior: "smooth"
			} );
		} );
		window.addEventListener( "scroll", function()
		{
			if( window.scrollY > 500 )
			{
				button.classList.add( "mathy-back-to-top--visible" );
			}
			else
			{
				button.classList.remove( "mathy-back-to-top--visible" );
			}
		} );
		document.body.appendChild( button );
	}
	
	function addCodeBlockLanguageBadges()
	{
		const codeBlocks = document.querySelectorAll( "pre > code[class*='language-']" );
		codeBlocks.forEach( function( codeBlock )
		{
			const pre = codeBlock.parentElement;
			if( !pre || pre.querySelector( ".mathy-code-badge" ) )
			{
				return;
			}
			const languageClass = Array.from( codeBlock.classList ).find( function( className )
			{
				return className.startsWith( "language-" );
			} );
			if( !languageClass )
			{
				return;
			}
			const language = languageClass.replace( "language-", "" ).trim();
			if( !language )
			{
				return;
			}
			pre.classList.add( "mathy-code-block" );
			const badge = createElement( "span", {
				className: "mathy-code-badge",
				text: language
			} );
			pre.appendChild( badge );
		} );
	}
	
	function getApiObjects()
	{
		const objects = Array.from( document.querySelectorAll( ".doc-object" ) );
		if( objects.length > 0 )
		{
			return objects;
		}
		return Array.from( document.querySelectorAll( "h2, h3, h4" ) ).filter( function( heading )
		{
			return heading.id && heading.closest( ".md-content__inner" );
		} );
	}
	
	function getObjectText( object )
	{
		return ( object.textContent || "" ).toLowerCase();
	}
	
	function addApiFilter()
	{
		if( !isApiPage() )
		{
			return;
		}
		const main = getMainContent();
		if( !main || main.querySelector( ".mathy-api-tools" ) )
		{
			return;
		}
		const tools = createElement( "section", {
			className: "mathy-api-tools",
			attributes: {
				"aria-label": "API page tools"
			}
		} );
		const title = createElement( "div", {
			className: "mathy-api-tools__title",
			text: "API Tools"
		} );
		const filter = createElement( "input", {
			className: "mathy-api-filter",
			attributes: {
				type: "search",
				placeholder: "Filter classes, methods, properties, or text...",
				"aria-label": "Filter API objects"
			}
		} );
		const status = createElement( "div", {
			className: "mathy-api-filter-status",
			text: ""
		} );
		tools.appendChild( title );
		tools.appendChild( filter );
		tools.appendChild( status );
		const firstHeading = main.querySelector( "h1" );
		if( firstHeading && firstHeading.nextSibling )
		{
			firstHeading.parentNode.insertBefore( tools, firstHeading.nextSibling );
		}
		else
		{
			main.insertBefore( tools, main.firstChild );
		}
		filter.addEventListener( "input", function()
		{
			const query = filter.value.trim().toLowerCase();
			const objects = getApiObjects();
			let visibleCount = 0;
			objects.forEach( function( object )
			{
				const matches = !query || getObjectText( object ).includes( query );
				object.classList.toggle( "mathy-api-object-hidden", !matches );
				if( matches )
				{
					visibleCount += 1;
				}
			} );
			if( !query )
			{
				status.textContent = "";
			}
			else
			{
				status.textContent = `${ visibleCount } matching API section${ visibleCount === 1
				                                                               ? ""
				                                                               : "s" }`;
			}
		} );
	}
	
	function addExpandCollapseControls()
	{
		if( !isApiPage() )
		{
			return;
		}
		const tools = document.querySelector( ".mathy-api-tools" );
		if( !tools || tools.querySelector( ".mathy-api-toggle-row" ) )
		{
			return;
		}
		const row = createElement( "div", {
			className: "mathy-api-toggle-row"
		} );
		const expand = createElement( "button", {
			className: "mathy-api-button",
			text: "Expand all",
			attributes: {
				type: "button"
			}
		} );
		const collapse = createElement( "button", {
			className: "mathy-api-button",
			text: "Collapse all",
			attributes: {
				type: "button"
			}
		} );
		expand.addEventListener( "click", function()
		{
			document.querySelectorAll( "details" ).forEach( function( details )
			{
				details.open = true;
			} );
		} );
		collapse.addEventListener( "click", function()
		{
			document.querySelectorAll( "details" ).forEach( function( details )
			{
				details.open = false;
			} );
		} );
		row.appendChild( expand );
		row.appendChild( collapse );
		tools.appendChild( row );
	}
	
	function normalizeHeadingText( text )
	{
		return ( text || "" )
				.replace( "¶", "" )
				.replace( "#", "" )
				.trim();
	}
	
	function inferImportPath( heading )
	{
		const text = normalizeHeadingText( heading.textContent );
		if( !text )
		{
			return "";
		}
		const moduleHeading = document.querySelector( "h1" );
		const moduleText = moduleHeading
		                   ? normalizeHeadingText( moduleHeading.textContent )
		                   : "";
		const moduleName = moduleText.toLowerCase().replace( /\s+/g, "_" );
		const match = text.match( /(?:class|def)?\s*([A-Za-z_][A-Za-z0-9_]*)/ );
		if( !match )
		{
			return "";
		}
		const symbol = match[ 1 ];
		if( !moduleName || symbol.toLowerCase() === moduleName )
		{
			return symbol;
		}
		return `from ${ moduleName } import ${ symbol }`;
	}
	
	function addCopyImportButtons()
	{
		if( !isApiPage() )
		{
			return;
		}
		const headings = document.querySelectorAll( ".doc-heading, h2, h3, h4" );
		headings.forEach( function( heading )
		{
			if( heading.querySelector( ".mathy-copy-import" ) )
			{
				return;
			}
			const importPath = inferImportPath( heading );
			if( !importPath || !importPath.includes( " import " ) )
			{
				return;
			}
			const button = createElement( "button", {
				className: "mathy-copy-import",
				text: "Copy import",
				attributes: {
					type: "button",
					title: importPath,
					"aria-label": `Copy import path ${ importPath }`
				}
			} );
			button.addEventListener( "click", async function( event )
			{
				event.preventDefault();
				event.stopPropagation();
				try
				{
					await navigator.clipboard.writeText( importPath );
					button.textContent = "Copied";
					setTimeout( function()
					{
						button.textContent = "Copy import";
					}, 1500 );
				}
				catch( error )
				{
					button.textContent = "Copy failed";
					setTimeout( function()
					{
						button.textContent = "Copy import";
					}, 1500 );
				}
			} );
			heading.appendChild( button );
		} );
	}
	
	function addDarkModeImageHelper()
	{
		const images = document.querySelectorAll( "img[data-mathy-light][data-mathy-dark]" );
		if( images.length === 0 )
		{
			return;
		}
		
		function isDarkMode()
		{
			const palette = document.body.getAttribute( "data-md-color-scheme" );
			const htmlPalette = document.documentElement.getAttribute( "data-md-color-scheme" );
			const scheme = palette || htmlPalette || "";
			if( scheme )
			{
				return scheme.toLowerCase().includes( "slate" ) ||
						scheme.toLowerCase().includes( "dark" );
			}
			return window.matchMedia && window.matchMedia( "(prefers-color-scheme: dark)" ).matches;
		}
		
		function updateImages()
		{
			images.forEach( function( image )
			{
				const darkSrc = image.getAttribute( "data-mathy-dark" );
				const lightSrc = image.getAttribute( "data-mathy-light" );
				image.setAttribute( "src", isDarkMode()
				                           ? darkSrc
				                           : lightSrc );
			} );
		}
		
		updateImages();
		const observer = new MutationObserver( updateImages );
		observer.observe( document.documentElement, {
			attributes: true,
			attributeFilter: [ "data-md-color-scheme" ]
		} );
		observer.observe( document.body, {
			attributes: true,
			attributeFilter: [ "data-md-color-scheme" ]
		} );
	}
	
	function initializeMathyDocs()
	{
		addBackToTopButton();
		addCodeBlockLanguageBadges();
		addApiFilter();
		addExpandCollapseControls();
		addCopyImportButtons();
		addDarkModeImageHelper();
	}
	
	onReady( initializeMathyDocs );
	if( window.document$ && typeof window.document$.subscribe === "function" )
	{
		window.document$.subscribe( function()
		{
			initializeMathyDocs();
		} );
	}
} )();